"""
    Implementation of Proximal Gradient on the BeaconEnv
"""

from rlsrc.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from rlsrc.mpi_tf import MpiAdamOptimizer, sync_all_params
from params import BeaconParams as BP
from rlsrc.logx import EpochLogger
import tensorflow as tf
import numpy as np
import scipy.signal
import json
import os

class Buffer:
    """
    A buffer for storing trajectories experienced by a PolicyGradient agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(Buffer.combined_shape(size, obs_dim), dtype=np.float32)
        # Actions buffer
        self.act_buf = np.zeros(size, dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # Rewards buffer
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Log probability of action a with the policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
        # ptr: Position to insert the next tuple
        # path_start_idx Posittion of the current trajectory
        # max_size Max size of the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    @staticmethod
    def discount_cumsum(x, discount):
        """
            x = [x0, x1, x2]
            output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    @staticmethod
    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew, logp):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Advantage
        self.adv_buf[path_slice] = Buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # Normalize the Advantage
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        return self.obs_buf, self.act_buf, self.adv_buf, self.logp_buf


class PolicyGradient(object):
    """
        Implementation of Proximal Policy Optimization
        This Implementation handle only continous values
    """
    def __init__(self, input_space, action_space, pi_lr, vf_lr, buffer_size, seed):
        super(PolicyGradient, self).__init__()
        # For serialize and unseialize
        self.initSavor = {'input_space':input_space,
                          'action_space':action_space,
                          'pi_lr':pi_lr,
                          'vf_lr':vf_lr,
                          'buffer_size':buffer_size,
                          'seed':seed}
        # Create a logger
        self.logger = EpochLogger(output_dir="./logger")
        self.logger.save_config(locals())
        # Stored the spaces
        self.input_space = input_space
        self.action_space = action_space
        self.seed = seed
        # PolicyGradient Buffer defined above
        self.buffer = Buffer(
            obs_dim=input_space,
            act_dim=action_space,
            size=buffer_size
        )
        # Learning rate of the policy network
        self.pi_lr = pi_lr
        # The tensorflow session (set later)
        self.sess = None
        # Apply a random seed on tensorflow and numpy
        tf.set_random_seed(BP.seed)
        np.random.seed(BP.seed)

    def compile(self):
        """
            Compile the model
        """
        # tf_map: Input: Input state
        # tf_adv: Input: Advantage
        self.tf_map, self.tf_a, self.tf_adv = PolicyGradient.inputs(
            map_space=self.input_space,
            action_space=self.action_space
        )
        # mu_op: Used to get the exploited prediction of the model
        # pi_op: Used to get the prediction of the model
        # logp_a_op: Used to get the log likelihood of taking action a with the current policy
        # logp_pi_op: Used to get the log likelihood of the predicted action @pi_op
        # log_std: Used to get the currently used log_std
        self.mu_op, self.pi_op, self.logp_a_op, self.logp_pi_op = PolicyGradient.mlp(
            tf_map=self.tf_map,
            tf_a=self.tf_a,
            action_space=self.action_space,
            seed=self.seed
        )
        # Error
        self.pi_loss = PolicyGradient.net_objectives(
            tf_adv=self.tf_adv,
            logp_a_op=self.logp_a_op
        )
        # Optimization
        self.train_pi = tf.train.AdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        # Entropy
        self.approx_ent = tf.reduce_mean(-self.logp_a_op)

    def set_sess(self, sess):
        # Set the tensorflow used to run this model
        self.sess = sess
        # Set up the saver
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.tf_map}, outputs={'pi': self.pi_op, "mu": self.mu_op,"logp_pi": self.logp_pi_op })

    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        mu, pi, logp_pi = self.sess.run([self.mu_op, self.pi_op, self.logp_pi_op], feed_dict={
            self.tf_map: states
        })
        return mu, pi, logp_pi

    def store(self, obs, act, rew, logp):
        # Store the observation, action, reward and the log probability of the action
        # into the buffer
        self.logger.store(Reward=rew)
        self.buffer.store(obs, act, rew, logp)

    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)

    def save(self, it):
        # Save model
        self.logger.log("Saving model. it=%s" % it)
        self.logger.save_state({}, it)
    
    def load(self, sess, model):
        # Load model
        saves = [int(x[11:]) for x in os.listdir("./logger") if model in x and len(x)>11]
        itr = '%d'%max(saves)
        print("Select this MODEL", itr)
        model = self.logger.load_state(sess, model, int(itr))
        self.mu_op = model['mu']
        self.pi_op = model['pi']
        self.logp_pi_op = model['logp_pi']
        

    def train(self, additional_infos={}):
        # Get buffer
        obs_buf, act_buf, adv_buf, logp_last_buf = self.buffer.get()
        # Train the model
        pi_loss_list = []
        entropy_list = []

        for step in range(5):
            _, entropy, pi_loss = self.sess.run([self.train_pi, self.approx_ent, self.pi_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_a:act_buf,
                self.tf_adv: adv_buf
            })

            pi_loss_list.append(pi_loss)
            entropy_list.append(entropy)

        self.logger.log_tabular("PiLoss", np.mean(pi_loss_list))
        self.logger.log_tabular("Entropy", np.mean(entropy_list))
        self.logger.log_tabular("Reward", average_only=True)
        for info in additional_infos:
            self.logger.log_tabular(info, additional_infos[info])
        self.logger.dump_tabular()

    @staticmethod
    def gaussian_likelihood(x, mu, log_std):
        # Compute the gaussian likelihood of x with a normal gaussian distribution of mean @mu
        # and a std @log_std
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @staticmethod
    def inputs(map_space, action_space):
        """
            @map_space Tuple of the space. Ex (size,)
            @action_space Tuple describing the action space. Ex (size,)
        """
        # Map of the game
        tf_map = tf.placeholder(tf.float32, shape=(None, *map_space), name="tf_map")
        # Possible actions (Should be two: x,y for the beacon game)
        tf_a = tf.placeholder(tf.int32, shape=(None,), name="tf_a")
        # Advantage
        tf_adv = tf.placeholder(tf.float32, shape=(None,), name="tf_adv")
        return tf_map, tf_a, tf_adv

    @staticmethod
    def mlp(tf_map, tf_a, action_space, seed=None):
        if seed is not None:
            tf.random.set_random_seed(seed)

        # Expand the dimension of the input
        tf_map_expand = tf.expand_dims(tf_map, axis=3)

        flatten = tf.layers.flatten(tf_map_expand)
        hidden = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)
        spacial_action_logits = tf.layers.dense(hidden, units=action_space, activation=None)

        # Add take the log of the softmax
        logp_all = tf.nn.log_softmax(spacial_action_logits)
        # Take random actions according to the logits (Exploration)
        pi_op = tf.squeeze(tf.multinomial(spacial_action_logits,1), axis=1)
        mu = tf.argmax(spacial_action_logits, axis=1)

        # Gives log probability, according to  the policy, of taking actions @a in states @x
        logp_a_op = tf.reduce_sum(tf.one_hot(tf_a, depth=action_space) * logp_all, axis=1)
        # Gives log probability, according to the policy, of the action sampled by pi.
        logp_pi_op = tf.reduce_sum(tf.one_hot(pi_op, depth=action_space) * logp_all, axis=1)

        return mu, pi_op, logp_a_op, logp_pi_op

    @staticmethod
    def net_objectives(logp_a_op, tf_adv, clip_ratio=0.2):
        """
            @v_op: Predicted value function
            @tf_tv: Expected advantage
            @logp_a_op: Log likelihood of taking action under the current policy
            @tf_logp_old_pi: Log likelihood of the last policy
            @tf_adv: Advantage input
        """
        pi_loss = -tf.reduce_mean(logp_a_op*tf_adv)
        return pi_loss
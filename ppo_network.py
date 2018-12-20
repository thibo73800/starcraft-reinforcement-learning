"""
    Implementation of Proximal Policy Network on the BeaconEnv
"""

from rlsrc.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from rlsrc.mpi_tf import MpiAdamOptimizer, sync_all_params
from params import BeaconParams as BP
from rlsrc.logx import EpochLogger
import tensorflow as tf
import numpy as np
import scipy.signal

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # Observations buffer
        self.obs_buf = np.zeros(PPOBuffer.combined_shape(size, obs_dim), dtype=np.float32)
        # Actions buffer
        self.act_buf = np.zeros(size, dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # Rewards buffer
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Expected returns Buffer
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # Values functions buffer
        self.val_buf = np.zeros(size, dtype=np.float32)
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

    def store(self, obs, act, rew, val, logp):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
            Call this at the end of a trajectory, or when one gets cut off
            by an epoch ending. This looks back in the buffer to where the
            trajectory started, and uses rewards and value estimates from
            the whole trajectory to compute advantage estimates with GAE-Lambda,
            as well as compute the rewards-to-go for each state, to use as
            the targets for the value function.
            The "last_val" argument should be 0 if the trajectory ended
            because the agent reached a terminal state (died), and otherwise
            should be V(s_T), the value function estimated for the last state.
            This allows us to bootstrap the reward-to-go calculation to account
            for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Append the last value function
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = PPOBuffer.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = PPOBuffer.discount_cumsum(rews, self.gamma)[:-1]
        # Set the new trajectory starting at the current ptr
        self.path_start_idx = self.ptr

    def get(self):
        """
            Call this at the end of an epoch to get all of the data from
            the buffer, with advantages appropriately normalized (shifted to have
            mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # MPI
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # Normalize the Advantage
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf

class PPO(object):
    """
        Implementation of Proximal Policy Optimization
        This Implementation handle only continous values
    """
    def __init__(self, input_space, action_space, pi_lr, vf_lr, buffer_size, seed):
        super(PPO, self).__init__()

        # Create a logger
        self.logger = EpochLogger(output_dir="./logger")
        self.logger.save_config(locals())
        # Stored the spaces
        self.input_space = input_space
        self.action_space = action_space
        self.seed = seed
        # PPO Buffer defined above
        self.buffer = PPOBuffer(
            obs_dim=input_space,
            act_dim=action_space,
            size=buffer_size
        )
        # Learning rate of the policy network and the value network
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr

        # The tensorflow session (set later)
        self.sess = None
        # Apply a random seed on tensorflow and numpy
        tf.set_random_seed(seed)
        np.random.seed(BP.seed)

    def compile(self):
        """
            Compile the model
        """
        # tf_map: Input: Input state
        # tf_adv: Input: Advantage
        # tf_tv: Input: Expected value function
        # tf_logp_old_pi : Log likelihood of the old policy. Used to track the KL divergence
        self.tf_map, self.tf_a, self.tf_adv, self.tf_tv, self.tf_logp_old_pi = PPO.inputs(
            map_space=self.input_space,
            action_space=self.action_space
        )
        # mu_op: Used to get the exploited prediction of the model
        # v_op : Used to get the predicted value of the input state
        # pi_op: Used to get the prediction of the model
        # logp_a_op: Used to get the log likelihood of taking action a with the current policy
        # logp_pi_op: Used to get the log likelihood of the predicted action @pi_op
        # log_std: Used to get the currently used log_std
        self.mu_op, self.v_op, self.pi_op, self.logp_a_op, self.logp_pi_op = PPO.fully_conv(
            tf_map=self.tf_map,
            tf_a=self.tf_a,
            action_space=self.action_space,
            seed=self.seed
        )
        # Error
        self.pi_loss, self.v_loss = PPO.net_objectives(
            v_op=self.v_op,
            logp_a_op=self.logp_a_op,
            tf_logp_old_pi=self.tf_logp_old_pi,
            tf_adv=self.tf_adv,
            tf_tv=self.tf_tv
        )
        # Kl divergence and Approximation of the entropy
        self.approx_kl, self.approx_ent = PPO.net_info(self.tf_logp_old_pi, self.logp_a_op)
        # Optimization
        # MPI
        #self.train_pi = MpiAdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        #self.train_v =  MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)
        self.train_pi = tf.train.AdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

    def set_sess(self, sess):
        # Set the tensorflow used to run this model
        self.sess = sess
        # Set up the saver
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.tf_map}, outputs={'pi': self.pi_op, 'v': self.v_op})

    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        mu, pi, logp_pi = self.sess.run([self.mu_op, self.pi_op, self.logp_pi_op], feed_dict={
            self.tf_map: states
        })
        return mu, pi, logp_pi

    def eval(self, states):
        # Evaluate the state
        val = self.sess.run([self.v_op], feed_dict={
            self.tf_map: states
        })
        return val

    def store(self, obs, act, rew, val, logp):
        # Store the observation, action, reward, value and the log probability of the action
        # into the buffer
        self.logger.store(Reward=rew)
        self.buffer.store(obs, act, rew, val, logp)

    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)

    def save(self, it):
        # Save model
        self.logger.log("Saving model. it=%s" % it)
        self.logger.save_state({}, it)

    def train(self, additional_infos={}):
        # Get buffer
        obs_buf, act_buf, adv_buf, value_target_buf, logp_last_buf = self.buffer.get()
        # Train the model
        pi_loss_list = []
        v_loss_list = []
        entropy_list = []

        for step in range(80):
            _, kl, entropy, pi_loss = self.sess.run([self.train_pi, self.approx_kl, self.approx_ent, self.pi_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_a:act_buf,
                self.tf_adv: adv_buf,
                self.tf_logp_old_pi: logp_last_buf,
                self.tf_tv: value_target_buf
            })

            pi_loss_list.append(pi_loss)
            entropy_list.append(entropy)

            if kl > 0.01 * 1.5:
                self.logger.log("Stop training pi du to max kl divergence at step :%s" % step)
                break

        for step in range(80):
            _, v_loss = self.sess.run([self.train_v, self.v_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_tv: value_target_buf
            })
            v_loss_list.append(v_loss)
            #print("v_loss", v_loss)
            #self.logger.store(VLoss=v_loss)

        # MPI
        #v_loss = mpi_avg(v_loss)
        #pi_loss = mpi_avg(pi_loss)
        #kl = mpi_avg(kl)

        self.logger.log_tabular("Kl", kl)
        self.logger.log_tabular("PiLoss", np.mean(pi_loss_list))
        self.logger.log_tabular("VLoss", np.mean(v_loss_list))
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
        # Target value function
        tf_tv = tf.placeholder(tf.float32, shape=(None,), name="tf_tv")
        # tf_logp_old_pi
        tf_logp_old_pi = tf.placeholder(tf.float32, shape=(None,), name="tf_logp_old_pi")

        return tf_map, tf_a, tf_adv, tf_tv, tf_logp_old_pi

    @staticmethod
    def fully_conv(tf_map, tf_a, action_space, seed=None):
        if seed is not None:
            tf.random.set_random_seed(seed)

        # Expand the dimension of the input
        tf_map_expand = tf.expand_dims(tf_map, axis=3)

        """
        # First convolution
        conv1 = tf.layers.conv2d(
            inputs=tf_map_expand,
            filters=8,
            kernel_size=[5, 5],
            padding="same", # Important to preserve space information
            activation=tf.nn.relu,
            strides=(1, 1)
        )

        # Second convolution
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[3, 3],
            padding="same", # Important to preserve space information
            activation=tf.nn.relu,
            strides=(1, 1)
        )
        #conv2_flatten = tf.layers.flatten(conv2)

        #conv2_flatten = tf.layers.flatten(conv1)

        # 1x1 Convolution
        spacial_action_logits = tf.layers.conv2d(
            inputs=conv1,
            filters=1,
            kernel_size=[1, 1],
            padding="same", # Important to preserve space information
            activation=None,
            strides=(1, 1)
        )

        spacial_action_logits = tf.layers.flatten(spacial_action_logits)
        """

        flatten = tf.layers.flatten(tf_map_expand)
        hidden = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)
        spacial_action_logits = tf.layers.dense(hidden, units=64*64, activation=None)

        # Add take the log of the softmax
        logp_all = tf.nn.log_softmax(spacial_action_logits)
        # Take random actions according to the logits (Exploration)
        pi_op = tf.squeeze(tf.multinomial(spacial_action_logits,1), axis=1)
        mu = tf.argmax(spacial_action_logits, axis=1)

        #log_std = tf.get_variable(name='log_std', initializer=-1.5*np.ones(action_space, dtype=np.float32))
        #std = tf.exp(log_std)
        #pi_op = mu + tf.random_normal(tf.shape(mu)) * std

        v_op = tf.layers.dense(inputs=hidden, units=1) # Value function

        # Gives log probability, according to  the policy, of taking actions @a in states @x
        #logp_a_op = PPO.gaussian_likelihood(tf_a, mu, log_std)
        logp_a_op = tf.reduce_sum(tf.one_hot(tf_a, depth=64*64) * logp_all, axis=1)
        # Gives log probability, according to the policy, of the action sampled by pi.
        #logp_pi_op = PPO.gaussian_likelihood(pi_op, mu, log_std)
        logp_pi_op = tf.reduce_sum(tf.one_hot(pi_op, depth=64*64) * logp_all, axis=1)

        return mu, v_op, pi_op, logp_a_op, logp_pi_op

    @staticmethod
    def net_objectives(v_op, logp_a_op, tf_logp_old_pi, tf_adv, tf_tv, clip_ratio=0.2):
        """
            @v_op: Predicted value function
            @tf_tv: Expected advantage
            @logp_a_op: Log likelihood of taking action under the current policy
            @tf_logp_old_pi: Log likelihood of the last policy
            @tf_adv: Advantage input
        """
        ratio = tf.exp(logp_a_op - tf_logp_old_pi) # pi(a|s) / pi_old(a|s)
        # Advantage
        min_adv = tf.where(tf_adv>0, (1+clip_ratio)*tf_adv, (1-clip_ratio)*tf_adv)
        # pi_loss
        minn = tf.minimum(ratio * tf_adv, min_adv)
        pi_loss = -tf.reduce_mean(minn)
        # v_loss
        v_loss = tf.reduce_mean((tf_tv - v_op)**2)
        return pi_loss, v_loss

    @staticmethod
    def net_info(tf_logp_old_pi, logp_a_op):
        # Approximation of the Kl CONVERGENCE
        approx_kl = tf.reduce_mean(tf_logp_old_pi - logp_a_op)
        approx_ent = tf.reduce_mean(-logp_a_op)
        return approx_kl, approx_ent

def main():
    # Number of state to start with
    NB = 2
    BUFFER_SIZE = 512

    # MPI
    # mpi_fork(1)

    # Create the PPO class
    ppo = PPO(
        input_space=BP.map_size,
        action_space=BP.action_space,
        pi_lr=BP.pi_lr,
        vf_lr=BP.vf_lr,
        buffer_size=BUFFER_SIZE,
        seed=BP.seed
    )
    ppo.compile()


    # Init Session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())
    # Sync params across processes
    sess.run(sync_all_params())
    # Set the session in ppo
    ppo.set_sess(sess)

    # Create the two state
    states = np.zeros((NB, *BP.map_size))
    for i in range(NB):
        states[i] += np.random.normal(0, 0.1, (BP.map_size)) + i/10
    # Set target value
    target = np.zeros((NB, 2))
    target[:int(NB/2)] = np.array([32, 64])
    target[int(NB/2):] = np.array([8, 16])

    # Step into the environment
    print("-----")
    mu, pi, _ = ppo.step(states)
    # Positions
    pirescale = np.expand_dims(mu, axis=1)
    pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
    positions = np.zeros_like(pirescale)
    positions[:,0] = pirescale[:,0] // 64
    positions[:,1] = pirescale[:,0] % 64

    print("Position", positions)
    diff = np.sqrt(np.square(np.subtract(target, positions)))
    reward = (-diff.sum(axis=1)/100)+0.05
    print("reward", reward)

    for epoch in range(100):
        print("epoch", epoch)
        added = 0
        while added < BUFFER_SIZE:
        #for step in range(BUFFER_SIZE):
            # Get a random state
            ii = np.random.randint(NB)
            n_state = states[ii]

            # Step with ppo according to this state
            mu, pi, last_logp_pi = ppo.step([n_state])
            # Positions
            pirescale = np.expand_dims(pi, axis=1)
            pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
            positions = np.zeros_like(pirescale)
            positions[:,0] = pirescale[:,0] // 64
            positions[:,1] = pirescale[:,0] % 64

            diff = np.sqrt(np.square(np.subtract(target[ii], positions)))
            reward = (-diff.sum(axis=1)/100)+0.01

            # Store the observation
            value = ppo.eval([n_state]) # Evalueate the given state
            #value = [0]
            if (reward[0] == 0):
                reward[0] = -1
            ppo.store(n_state, pi, reward[0], value[0], last_logp_pi)
            added += 1

            ppo.finish_path(value[0])

        ppo.train()

        print("-----")
        mu, pi, _ = ppo.step(states)
        # Positions
        pirescale = np.expand_dims(mu, axis=1)
        pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
        positions = np.zeros_like(pirescale)
        positions[:,0] = pirescale[:,0] // 64
        positions[:,1] = pirescale[:,0] % 64
        print(positions)


if __name__ == '__main__':
    main()

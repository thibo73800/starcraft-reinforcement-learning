from rlsrc.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from rlsrc.mpi_tf import MpiAdamOptimizer, sync_all_params
from  rlsrc.run_utils import setup_logger_kwargs
from rlsrc.logx import restore_tf_graph
from params import BeaconParams as BP
from parent_agent import ParentAgent
from rlsrc.logger import Logger
import beacon_network as network

from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
import matplotlib.pyplot as plt
from pysc2.env import sc2_env
from docopt import docopt
import tensorflow as tf
from absl import app, flags
import numpy as np
import argparse
import random
import math
import sys
import os


log = Logger("Beacon")

# Features screen
FS_PLAYER_ID = features.SCREEN_FEATURES.player_id
FS_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative
FS_SELECTED = features.SCREEN_FEATURES.selected
FS_ENERGY = features.SCREEN_FEATURES.unit_energy
FS_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points

# define contstants for actions
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
# constants for actions
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

class Agent(ParentAgent):
	"""
		An agent for doing a simple movement form one point to another.
	"""

	def __init__(self, model=None):
		super(Agent, self).__init__()
		self.model = model
		self.nb_steps = 0
		self.max_steps = 512
		self.epoch = 0

		seed = BP.seed + 10000 * proc_id()

		self.local_steps_per_epoch = int(self.max_steps / num_procs())

		# Create the NET class
		self.policyGradient = network.PolicyGradient(
			input_space=BP.map_size,
			action_space=BP.action_space,
			pi_lr=BP.pi_lr,
			vf_lr=BP.vf_lr,
			buffer_size=self.local_steps_per_epoch,
			seed=BP.seed
		)
		self.policyGradient.compile()
        # Init Session
		sess = tf.Session()
		# Init variables
		sess.run(tf.global_variables_initializer())
		# Sync params across processes
		sess.run(sync_all_params())
		if model is not None:
			self.policyGradient.load(sess,'simple_save')
			saves = [int(x[11:]) for x in os.listdir("./logger") if model in x and len(x)>11]
			itr = '%d'%max(saves)
			self.epoch = int(itr)
        
		# Set the session in ppo
		self.policyGradient.set_sess(sess)

		self.current_tuple = None

            
	def trainAgent(self,obs_new, obs, action, reward,last_logp_pi):
		#
		reward = -1 if reward == 0 else 1
		obs = self.get_feature_screen(obs, FS_PLAYER_RELATIVE)
        
		self.policyGradient.store(obs, action, reward, last_logp_pi)
		# Increase the current step
		self.nb_steps += 1
		# Finish the episode on reward == 1
		if reward == 1 and self.nb_steps != self.local_steps_per_epoch and not obs_new.last():
			self.policyGradient.finish_path(reward)
		# If this is the end of the epoch or this is the last observation
		if self.nb_steps == self.local_steps_per_epoch or obs_new.last():
			# Retrieve the features
			features = self.get_feature_screen(obs_new, FS_PLAYER_RELATIVE)
			# If this is the last observation, we bootstrap the value function
			self.policyGradient.finish_path(obs_new.reward)

			# We do not train yet if this is just the end of the current episode
			if obs_new.last() is True and self.nb_steps != self.local_steps_per_epoch:
				return

			self.policyGradient.train({"Epoch": self.epoch})

			self.nb_steps = 0
			self.epoch += 1
			# Save every 100 epochs
			if (self.epoch-1) % 300 == 0:
				self.policyGradient.save(self.epoch)
    
    
	def predictionToPosition(self,pi, dim = 64):
		pirescale = np.expand_dims(pi, axis=1)
		pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
		positions = np.zeros_like(pirescale)
		positions[:,0] = pirescale[:,0] // dim
		positions[:,1] = pirescale[:,0] % dim
		return positions

	def step(self, obs):
		# step function gets called automatically by pysc2 environment
		# call the parent class to have pysc2 setup rewards/etc for u
		super(Agent, self).step(obs)
		# if we can move our army (we have something selected)
		if _MOVE_SCREEN in obs.observation['available_actions']:

			# Get the features of the screen
			features = self.get_feature_screen(obs, FS_PLAYER_RELATIVE)
        	# Step with ppo according to this state
			mu, pi, last_logp_pi = self.policyGradient.step([features])
			# Convert the prediction into positions
			positions = self.predictionToPosition(pi)
			# Create the next tueple
			self.current_tuple = [features, pi[0], None, last_logp_pi]
			# Get a random location on the map
			return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, positions[0]]) , pi[0] ,last_logp_pi

		# if we can't move, we havent selected our army, so selecto ur army
		else:
			return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]), None, None

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='Name of the model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--training', type=bool, help="if is training")
	args, unknown_flags = parser.parse_known_args()

	model = args.model
	visualize = False #
	replay = args.replay #
	isTraining = args.training

	step_mul = 16 if model is None else 16
	save_replay_episodes = 10 if replay else 0

	agent = Agent(model=model)

	mpi_fork(BP.cpu)  # run parallel code with mpi
	# Get the path to the 'output_dir' and 'exp_name' for the given seed
	logger_kwargs = setup_logger_kwargs(BP.exp_name, BP.seed)

	try:
		with sc2_env.SC2Env(map_name="MoveToBeacon", players=[sc2_env.Agent(sc2_env.Race.zerg)], agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=BP.map_size[0], minimap=64),
			use_feature_units=True),
			step_mul=step_mul, # Number of step before to ask the next action to from the agent
			visualize=visualize,
			save_replay_episodes=save_replay_episodes,
			replay_dir=os.path.dirname(os.path.abspath(__file__)),
			) as env:

			for i in range(100000):
				agent.setup(env.observation_spec(), env.action_spec())
				timesteps = env.reset()
				agent.reset()

				while True:
					action, pi, last_logp_pi = agent.step(timesteps[0])
					step_actions = [action]
					old_timesteps = timesteps
					timesteps = env.step(step_actions)
					if(isTraining):
						agent.trainAgent(timesteps[0], old_timesteps[0], pi, timesteps[0].reward, last_logp_pi)
					if timesteps[0].last():
						break

	except KeyboardInterrupt:
		pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='Name of the model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--training', type=bool, help="if is training")
	args, unknown_flags = parser.parse_known_args()
	flags.FLAGS(sys.argv[:1] + unknown_flags)

	app.run(main, argv=sys.argv[:1] + unknown_flags)

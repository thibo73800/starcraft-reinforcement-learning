"""
Beacon.

Usage:
  beacon.py [--model=<name>] [--replay]

Options:
  -h --help     		Show this screen.
  --model=<name>     	Name of the model to load
  --replay  			Used to save a replay
"""

from rlsrc.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from rlsrc.mpi_tf import MpiAdamOptimizer, sync_all_params
from  rlsrc.run_utils import setup_logger_kwargs
from rlsrc.logx import restore_tf_graph
from params import BeaconParams as BP
from parent_agent import ParentAgent
from rlsrc.logger import Logger
import ppo_network as network

from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
import matplotlib.pyplot as plt
from pysc2.env import sc2_env
from docopt import docopt
import tensorflow as tf
from absl import app
import numpy as np
import random
import math
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
		if model is not None: # Load model
			saves = [int(x[11:]) for x in os.listdir("./logger") if model in x and len(x)>11]
			itr = '%d'%max(saves)
			print("Select this MODEL", itr)
			# load the things!
			sess = tf.Session()
			model = restore_tf_graph(sess, os.path.join("./logger", model+itr))
			action_op = model['pi']
			# make function for producing an action given a single state
			self.get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})
		else: # Create the model
			self.nb_steps = 0
			self.max_steps = 512
			self.epoch = 0

			seed = BP.seed + 10000 * proc_id()

			self.local_steps_per_epoch = int(self.max_steps / num_procs())

			# Create the PPO class
			self.ppo = network.PPO(
				input_space=BP.map_size,
				action_space=BP.action_space,
				pi_lr=BP.pi_lr,
				vf_lr=BP.vf_lr,
				buffer_size=self.local_steps_per_epoch,
				seed=BP.seed
			)
			self.ppo.compile()

			# Init Session
			sess = tf.Session()
			# Init variables
			sess.run(tf.global_variables_initializer())
			# Sync params across processes
			sess.run(sync_all_params())
			# Set the session in ppo
			self.ppo.set_sess(sess)

			self.current_tuple = None

	def handle_tuple(self, obs):
		# Handle the current tuple
		if self.current_tuple is not None:
			self.current_tuple[2] = -1 if obs.reward == 0 else 1
			t = self.current_tuple
			self.ppo.store(t[0], t[1], t[2], t[3], t[4])
			# Increase the current step
			self.nb_steps += 1
			# Finish the episode on reward == 1
			if obs.reward == 1 and self.nb_steps != self.local_steps_per_epoch and not obs.last():
				self.ppo.finish_path(obs.reward)
			# If this is the end of the epoch or this is the last observation
			if self.nb_steps == self.local_steps_per_epoch or obs.last():
				# Retrieve the features
				features = self.get_feature_screen(obs, FS_PLAYER_RELATIVE)
				# If this is the last observation, we bootstrap the value function

				value = self.ppo.eval([features])[0][0][0]
				self.ppo.finish_path(value)

				# We do not train yet if this is just the end of the current episode
				if obs.last() is True and self.nb_steps != self.local_steps_per_epoch:
					return

				self.ppo.train({"Epoch": self.epoch})

				self.nb_steps = 0
				self.epoch += 1
				# Save every 100 epochs
				if (self.epoch-1) % 100 == 0:
					self.ppo.save(self.epoch)

			self.current_tuple = None

	def move_screen(self, obs):
		# Store the tuple in memory
		self.handle_tuple(obs)
		# Get the features of the screen
		features = self.get_feature_screen(obs, FS_PLAYER_RELATIVE)
		# Step with ppo according to this state
		mu, pi, last_logp_pi = self.ppo.step([features])
		# Convert the prediction into positions
		pirescale = np.expand_dims(pi, axis=1)
		pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
		positions = np.zeros_like(pirescale)
		positions[:,0] = pirescale[:,0] // 64
		positions[:,1] = pirescale[:,0] % 64

		# Evaluate the given state
		value = self.ppo.eval([features])[0][0][0]
		value = 0

		# Create the next tueple
		self.current_tuple = [features, pi[0], None, value, last_logp_pi]

		# Get a random location on the map
		return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, positions[0]])

	def step(self, obs):
		# step function gets called automatically by pysc2 environment
		# call the parent class to have pysc2 setup rewards/etc for u
		super(Agent, self).step(obs)
		# if we can move our army (we have something selected)
		if _MOVE_SCREEN in obs.observation['available_actions']:
			if not self.model:
				return self.move_screen(obs)
			else:
				features = self.get_feature_screen(obs, FS_PLAYER_RELATIVE)

				pi = self.get_action(features)
				pirescale = np.expand_dims(pi, axis=1)
				pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
				positions = np.zeros_like(pirescale)
				positions[:,0] = pirescale[:,0] // 64
				positions[:,1] = pirescale[:,0] % 64

				return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED,  positions[0]])
		# if we can't move, we havent selected our army, so selecto ur army
		else:
			return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

def main(_):
	model = "simple_save"
	replay = False
	visualize = False

	step_mul = 128 if model is None else 16
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
			replay_dir="/home/thibault/work/rl/starcraft2",
			) as env:

			for i in range(100000):
				agent.setup(env.observation_spec(), env.action_spec())
				timesteps = env.reset()
				agent.reset()

				while True:
					step_actions = [agent.step(timesteps[0])]
					if timesteps[0].last():
						break
					timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
	app.run(main)

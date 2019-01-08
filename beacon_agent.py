from rlsrc.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from rlsrc.mpi_tf import MpiAdamOptimizer, sync_all_params
from  rlsrc.run_utils import setup_logger_kwargs
from params import BeaconParams as BP
from parent_agent import ParentAgent
from rlsrc.logger import Logger
import ppo_network as network

from pysc2.lib import actions, features, units
from rlsrc.logx import restore_tf_graph
from pysc2.agents import base_agent
import matplotlib.pyplot as plt
from pysc2.env import sc2_env
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

class SmartAgent(ParentAgent):
    """
        An agent for doing a simple movement form one point to another
    """
    def __init__(self):
        super(SmartAgent, self).__init__()

        saves = [int(x[11:]) for x in os.listdir("./logger") if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves)
        print("Select this MODEL", itr)
        # load the things!
        sess = tf.Session()
        model = restore_tf_graph(sess, os.path.join("./logger", 'simple_save'+itr))
        action_op = model['pi']
        # make function for producing an action given a single state
        self.get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})

    def step(self, obs):
        # step function gets called automatically by pysc2 environment
        # call the parent class to have pysc2 setup rewards/etc for u
        super(SmartAgent, self).step(obs)
        # if we can move our army (we have something selected)
        if _MOVE_SCREEN in obs.observation['available_actions']:
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

def main(unused_argv):
	agent = SmartAgent()

	mpi_fork(BP.cpu)  # run parallel code with mpi
	# Get the path to the 'output_dir' and 'exp_name' for the given seed
	logger_kwargs = setup_logger_kwargs(BP.exp_name, BP.seed)

	try:
		with sc2_env.SC2Env(map_name="MoveToBeacon", players=[sc2_env.Agent(sc2_env.Race.zerg)], agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=BP.map_size[0], minimap=64),
			use_feature_units=True),
			step_mul=16, # Number of step before to ask the next action to from the agent
			visualize=True,
			save_replay_episodes=1,
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

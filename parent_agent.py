import matplotlib.pyplot as plt
from pysc2.agents import base_agent
import numpy as np

class ParentAgent(base_agent.BaseAgent):
    """
        ParentAgent class with usefull methods for other agents
    """

    def __init__(self):
        # Init the BaseAgent
        super(ParentAgent, self).__init__()

    def get_feature_screen(self, obs, screen_feature):
    	# Get the feature associated with the observation
    	mapp = obs.observation["feature_screen"][screen_feature.index]
    	return np.array(mapp)

    def plot_feature_screen(self, obs, screen_feature):
    	# Get the feature associated with the observation
    	mapp = obs.observation["feature_screen"][screen_feature.index]
    	# Floor point (in case there is flaot points)
    	mapp = np.floor(np.array(mapp)).astype(int)
    	# Create the image to display
    	n_img = np.zeros((mapp.shape[0], mapp.shape[1], 3))
    	# Change the color of the image for each value in the scale of possible values
    	for index in range(screen_feature.scale):
    		n_img[mapp==index] = screen_feature.palette[index]
    	# Plot image
    	plt.imshow(n_img)
    	plt.show()


from plato_copilot.utils.log_utils import get_copilot_logger

logger = get_copilot_logger()

class Copilot():
	def __init__():
        # For functions with not implemented error, they must be implemented in the derived class
        pass

	def observe(self, obs):
		# Take the observation of the system, it could be
		# - some physical states
		# - images
		# - or any other sensory data
        raise NotImplementedError
	
	def preprocess_obs(self):
		# preprocess observation into desired object states, if the observations are images. 
        pass

	def assitive_decision(self, horizon=None):
		# The robot suggests a bunch of actions to follow
		# It can be any implementation (or even human inputs)
		# A solution for autonomous decision is sampling-based approach, primitive based
        raise NotImplementedError

	def forward_dynamics(self, action_seq):
        # Given a batch of action sequence, return a batch of predicted future states
        # Need to define maximal horizon, and the horizon for forwarding dynamics is
        # by default the same as the steps of action_sequence.
        # We probably also need to define some trivial action so that 
        # different action sequences are padded to same length
        raise NotImplementedError


	def visualize(self, predicted_state_seq):
		# The copilot system also needs to define this function to visualize 
		# the predicted state sequence so that human users can understand the 
		# action output
        raise NotImplementedError

class PlatoCopilot():
	def __init__(self):
		# This PlatoCopilot is designed for playing with Jenga Tower
		self.game_configuraiton = {}
		# initialize jenga towers
		self.jenga_states = None
		self.jenga_state_estimator = JengaStateEstimator()
		self.learned_jenga_dynamics = JengaDynamicModel().load(checkpoint_path)
        logger.debug("Initializing the copilot. Base initialization")

	def observe(self, obs):
		images = obs["images"]
		jenga_states = self.jenga_state_estimator(image)
		# pre-interaction data
		self.pre_interaction_states, self.pre_interaction_actions = self.pre_interaction()
		
	def assitive_decision(self, horizon=None):
	
		cofirmed_execution = False
		
		while not confirmed_execution:
		  # Ask human user to specify which block or what actions to take
			# 1a. sample faces from the tracked jenga block poses.
			# 1b. prompt users with faces / or interaction points
            logger.debug("Prompt user with possible Jenga blocks.")
			prompt_user_with_possible_interaction_points()
			
            logger.debug("Sampled action trajectory and predicted block states.")

			# 2a. sample actions, in shape [B, T, D]
			# B: Batch size, T: Time horizon, D: Action dimensions
			sampled_action_trajectory = self.sample_actions()
			predicted_block_states = self.forward_dynamics(sampled_action_trajectory)
			
			# 2b. prompt users with possible actions to select from, with suggested actions 
			prompt_user_with_possible_actions()
			# 2c. visualize the effect of any selected action
			self.visualize(predicted_block_states)
			
			# get confirmation of execution
			confirmed_execution = self.confirm_action(selected_actions)
		
	def forward_dynamics(self, batched_action_seq):
	  predicted_block_states = self.learned_jenga_dynamics(
	                                                       batched_action_seq, 
	                                                       self.pre_interaction_states, 
	                                                       self.pre_interaction_actions)
	  return predicted_block_states
	  

	def visualize(self, predicted_state_seq):
		# The copilot system also needs to define this function to visualize 
		# the predicted state sequence so that human users can understand the 
		# action output
        pass
"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env

from robosuite.wrappers import Wrapper

class ControlDelta:
    def __init__ (self, action, env):
        self.action=action
        # self.state = state
        self.position_limits=0.02
        self.env= env
        self.output_max = self.env.robots[0].controller.output_max
        self.output_min = self.env.robots[0].controller.output_min
        self.input_max = self.env.robots[0].controller.input_max
        self.input_min = self.env.robots[0].controller.input_min
        self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        self.action_output_transform = (self.output_max + self.output_min) / 2.0
        self.action_input_transform = (self.input_max + self.input_min) / 2.0
    
    def rescale_agent_delta(self) -> np.ndarray:
        action = np.clip(self.action[-3:], self.input_min, self.input_max)
        self.transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform
        return self.transformed_action
    
class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed
        If the table height is not varied then the while loop need not be executed and the robot will be reset to the initial position
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        
        site_0 = self.env.objs[0].sites[0]
        '''
        # The below code is the subroutine to make a gentle contact with the robot without changing its initial reset position
        
        site_pos_0 = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(site_0)]  
        dist=np.inf     
        action= np.zeros(self.env.action_spec[0].shape)
        position_limits=0.03
        indent=0.01
        
        while dist>0.003:
            eef_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
            action[:12] = self.env.robots[0].controller.guide_policy_gains
            action[12:14] = eef_pos[:2] + np.clip(site_pos_0[:2]-eef_pos[:2], a_min=np.array([-position_limits, -position_limits]), a_max=np.array([position_limits, position_limits]))
            action[-1] = eef_pos[-1] + np.clip(site_pos_0[-1] -  eef_pos[-1] - indent, a_min = -position_limits, a_max=position_limits) 
            self.env.step(action)

            dist = eef_pos[2]-site_pos_0[2] #just check the height
            # print(dist)
            if dist<0.07:
                action[12:]=eef_pos
                self.env.step(action)
        '''
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        eef_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        action[-3:] = ControlDelta(action,self).rescale_agent_delta()
        action[-3:] = eef_pos + np.clip(action[-3:], a_min=np.ones(3) * (-self.position_limits),\
                            a_max=np.ones(3) * (self.position_limits))
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

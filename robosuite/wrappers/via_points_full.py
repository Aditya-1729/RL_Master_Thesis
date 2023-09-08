"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env
from robosuite.wrappers import Wrapper
from itertools import cycle

class Via_points_full(Wrapper, gym.Env):
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

    def __init__(self, env, cfg, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__
        self.position_limits = cfg.task_config.clip
        #configuration parameters
        self.indent = cfg.task_config.indent
        # self.agent_config = cfg.agent_config
        self.kp = cfg.controller.kp
        self.kd = cfg.controller.damping_ratio
        self.dist_th = cfg.task_config.dist_th
        # self.site_pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.env.objs[0].sites[8])] 
        # Get reward range
        self.reward_range = (0, self.env.reward_scale)
        self.sites = cycle(self.env.objs[0].sites)
        self.site = self.env.objs[0].sites[0]
        self.site_pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.site)]


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

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        self.sites = cycle(self.env.objs[0].sites)
        self.site = self.env.objs[0].sites[0]
        self.site_pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.site)] 
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
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
        # Adding nominal control stuffs
        '''
        Modify the action command by forcing a nominal controller to output positions
        
        '''

        # action = np.array((3,3,3,3,3,3, 200,200,200,200,200,200))
        eef_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        # print(f"pre: {action}")
        # if self.agent_config==2:
        a = np.empty(15)
        a[:12]=action
        a[12:14] = eef_pos[:2] + np.clip(self.site_pos[:2]-eef_pos[:2], a_min=np.array([-self.position_limits, -self.position_limits]), a_max=np.array([self.position_limits, self.position_limits]))
        a[-1] = eef_pos[-1] + np.clip(self.site_pos[-1] - self.indent - eef_pos[-1], a_min = -self.position_limits, a_max=self.position_limits)
        # a[12:14] = self.site_pos[:2]
        # a[-1] = self.site_pos[-1] - self.indent
        delta = eef_pos - self.site_pos
        dist = np.linalg.norm(delta)
        # print(f"dist:{dist} site: {self.site}")
        if dist < self.dist_th:
            self.site = next(self.sites)
            self.site_pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.site)]
        # print(f"post: {a}")
        # if self.agent_config==3:


        ob_dict, reward, terminated, info = self.env.step(a)
        # print(f"reward:{reward}")

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
    
    # def process_action(self, pre_action):
    #     if self.agent_config==2:


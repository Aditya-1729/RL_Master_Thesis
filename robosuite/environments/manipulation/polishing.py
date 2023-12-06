import random
from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import PolishingArena
# from robosuite.models.objects import RoundobjObject, SquareobjObject
from robosuite.models.objects import Flat_top, Incline
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

DEFAULT_WIPE_CONFIG = {
    # settings for reward
    "arm_limit_collision_penalty": -10.0,
    # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
    "wipe_contact_reward": 0.5,  # reward for contacting something with the wiping tool  #0.1
    "unit_wiped_reward": 50.0,  # reward per peg wiped
    "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.5,  # penalty for each step that the force is over the safety threshold       #0.05
    "distance_multiplier": 10.0,
    # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
    "distance_th_multiplier": 5.0,  # multiplier in the tanh function for the aforementioned reward
    # settings for table top
    "table_full_size": [0.5, 0.8, 0.05],  # Size of tabletop
    "table_offset": [0.15, 0, 0.9],  # Offset of table (z dimension defines max height of table)
    "table_friction": [0.03, 0.005, 0.0001],  # Friction parameters for the table
    "table_friction_std": 0,  # Standard deviation to sample different friction parameters for the table each episode
    "table_height": 0.0,  # Additional height of the table over the default location
    "table_height_std": 0.0,  # Standard deviation to sample different heigths of the table each episode
    "line_width": 0.04,  # Width of the line to wipe (diameter of the pegs)
    "two_clusters": False,  # if the dirt to wipe is one continuous line or two
    "coverage_factor": 0.6,  # how much of the table surface we cover
    "num_markers": 100,  # How many particles of dirt to generate in the environment
    # settings for thresholds
    "contact_threshold": 0.30,  # Minimum eef force to qualify as contact [N]
    "pressure_threshold": 2,  # force threshold (N) to overcome to get increased contact wiping reward
    "pressure_threshold_max": 5.0,  # maximum force allowed (N)
    "target_force": 3.5,
    # misc settings
    "print_results": False,  # Whether to print results or not
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": False,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,  # Whether we allow for early terminations or not
    "use_condensed_obj_obs": False,
    # Whether to use condensed object observation representation (only applicable if obj obs is active)
    "no_contact_penalty": 0,
    "force_multiplier": 0.5,
    "reward_mode": 0,

}


def _compute_penalty(value, target_value, lower_limit, upper_limit):
    """ Compute the deviation of value and target_value in percent.
    Given an lower_limit and an upper_limit.

    lower_limit -------- target ----x------------ upper_limit
    --> x is the current value.

    Deviations between lower_limit and target is scaled to [0,1]
    Deviations between upper_limit and target is scaled to [0,1]
    Deviations above or below the limits are 1.

    :param value: Current Value (float)
    :param target_value: Target Value (float)
    :param lower_limit: Lower interval boundary (float)
    :param upper_limit: Upper interval boundary (float)
    :return: relative penalty [0,1] clipped to boundaries
    """

    penalty = 0
    delta_value = np.abs(value - target_value)

    if value > target_value:  # we are exceeding the target value
        penalty = delta_value / np.abs(upper_limit - target_value)

    if value < target_value:  # we are below the target value
        penalty = delta_value / np.abs(target_value - lower_limit)

    if penalty > 1:  # If outside [lower_limit, upper_limit] give maximum penalty
        penalty = 1

    return penalty


class Polishing(SingleArmEnv):
    """
    This class corresponds to the Wiping task for a single robot arm

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory.
            For this environment, setting a value other than the default ("WipingGripper") will raise an
            AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
            parameters, see the default configuration dict at the top of this file.
            If None is specified, the default configuration will be used.

        Raises:
            AssertionError: [Gripper specified]
            AssertionError: [Bad reward specification]
            AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="CustomGripper",
            initialization_noise=None,  # "default"
            use_camera_obs=True,
            use_object_obs=False,
            reward_scale=1.0,
            reward_shaping=True,
            placement_initializer=None,
            single_object_mode=2,  # mode single to randomly select one of the two objects on reset
            obj_type="Flat_top",
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1500,
            _max_episode_steps=100,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,  # {None, instance, class, element}
            task_config=None,
            renderer="mujoco",
            renderer_config=None,
    ):
        # Assert that the gripper type is None
        # assert (
        #     gripper_types == "WipingGripper"
        # ), "Tried to specify gripper other than WipingGripper in Wipe environment!"

        # Get config
        self.joint_limits = 0
        self.reward_done = None
        self.reward_contact = None
        self.penalty_yvel = None
        self.penalty_xdist = None
        self.penalty_force = None
        self.penalty_xvel = None
        self.task_config = task_config if task_config is not None else DEFAULT_WIPE_CONFIG

        # adding this because the wrapper threw an error for getting attribute
        # self._max_episode_steps = _max_episode_steps

        # settings for the reward
        self.total_force_ee = 0
        # self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config["arm_limit_collision_penalty"]
        # self.wipe_contact_reward = self.task_config["wipe_contact_reward"]
        # self.unit_wiped_reward = self.task_config["unit_wiped_reward"]
        # self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        # self.excess_force_penalty_mul = self.task_config["excess_force_penalty_mul"]
        # self.distance_multiplier = self.task_config["distance_multiplier"]
        # self.distance_th_multiplier = self.task_config["distance_th_multiplier"]
        # self.force_multiplier = self.task_config["force_multiplier"]
        # self.target_force = self.task_config["target_force"]
        # self.general_penalty = self.task_config["general_penalty"]
        # self.position_limits = self.task_config["clip"]
        self.dist_th = self.task_config["dist_th"]

        # Reward related Config -------------
        self.target_force = self.task_config["target_force"]
        self.reward_calc_upper_limit_force = self.task_config["reward_calc_upper_limit_force"]  # Missing in config
        self.reward_calc_lower_limit_force = self.task_config["reward_calc_lower_limit_force"]  # Missing in config
        assert (self.target_force < self.reward_calc_upper_limit_force)
        assert (self.target_force > self.reward_calc_lower_limit_force)

        self.target_xvel = self.task_config["target_xvel"]  # Missing in config
        self.reward_calc_upper_limit_xvel = self.task_config["reward_calc_upper_limit_xvel"]  # Missing in config
        self.reward_calc_lower_limit_xvel = self.task_config["reward_calc_lower_limit_xvel"]  # Missing in config
        assert (self.target_xvel < self.reward_calc_upper_limit_xvel)
        assert (self.target_xvel > self.reward_calc_lower_limit_xvel)

        self.target_xdist = self.task_config["target_xdist"]
        self.reward_calc_upper_limit_xdist = self.task_config["reward_calc_upper_limit_xdist"]  # Missing in config
        self.reward_calc_lower_limit_xdist = self.task_config["reward_calc_lower_limit_xdist"]  # Missing in config
        assert (self.reward_calc_lower_limit_xdist < self.reward_calc_upper_limit_xdist)

        self.target_yvel = self.task_config["target_yvel"]  # Missing in config
        self.reward_calc_upper_limit_yvel = self.task_config["reward_calc_upper_limit_yvel"]  # Missing in config
        self.reward_calc_lower_limit_yvel = self.task_config["reward_calc_lower_limit_yvel"]  # Missing in config
        assert (self.target_yvel < self.reward_calc_upper_limit_yvel)
        assert (self.target_yvel > self.reward_calc_lower_limit_yvel)

        self.reward_calc_min_force_for_contact = self.task_config["reward_calc_min_force_for_contact"]
        assert (self.reward_calc_min_force_for_contact < self.target_force)

        self.reward_calc_c_force = self.task_config["reward_calc_c_force"]
        self.reward_calc_c_xvel = self.task_config["reward_calc_c_xvel"]
        self.reward_calc_c_xdist = self.task_config["reward_calc_c_xdist"]
        self.reward_calc_c_yvel = self.task_config["reward_calc_c_yvel"]
        self.reward_calc_c_contact = self.task_config["reward_calc_c_contact"]
        self.reward_calc_c_done = self.task_config["reward_calc_c_done"]
        # ------------- Reward related config

        # Reward for maintaining force in the right window
        self.reward_mode = self.task_config["reward_mode"]
        # self.force_reward = self.force_multiplier*self.target_force

        # vel_threshold
        self.min_vel = self.task_config["min_vel"]
        self.max_vel = self.task_config["max_vel"]

        # position tracking
        #self.position_track_multiplier = self.task_config["position_track_multiplier"]
        # Final reward computation
        # So that is better to finish that to stay touching the table for 100 steps
        # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
        # self.task_complete_reward = self.unit_wiped_reward * (self.wipe_contact_reward + 0.5)
        #self.task_complete_reward = self.task_config["task_complete_reward"]
        # Verify that the distance multiplier is not greater than the task complete reward
        #assert (
        #        self.task_complete_reward > self.distance_multiplier
        #), "Distance multiplier cannot be greater than task complete reward!"

        # settings for table top

        self.table_full_size = self.task_config["table_full_size"]
        self.table_height = self.task_config["table_height"]
        self.table_height_std = self.task_config["table_height_std"]
        delta_height = min(0, np.random.normal(self.table_height, self.table_height_std))  # sample variation in height
        self.table_offset = np.array(self.task_config["table_offset"]) + np.array((0, 0, delta_height))
        self.table_friction = self.task_config["table_friction"]
        self.table_friction_std = self.task_config["table_friction_std"]
        self.line_width = self.task_config["line_width"]
        self.two_clusters = self.task_config["two_clusters"]
        self.coverage_factor = self.task_config["coverage_factor"]
        # self.num_markers = self.task_config["num_markers"] #TODO
        self.num_markers = 8  # len(self.objs[0].sites[:-1])      #it wont work like this

        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold = self.task_config["pressure_threshold"]
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]
        self.f_cap = self.task_config["f_safe"]
        # misc settings
        # self.use_force_obs = self.task_config["use_force_obs"]
        self.use_force_obs = True
        self.print_results = self.task_config["print_results"]
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.use_contact_obs = self.task_config["use_contact_obs"]
        self.early_terminations = self.task_config["early_terminations"]
        self.use_condensed_obj_obs = self.task_config["use_condensed_obj_obs"]
        self.horizon = horizon
        # Scale reward if desired (see reward method for details)
        # self.reward_normalization_factor = horizon / (
        #     self.num_markers * self.unit_wiped_reward + horizon * (self.wipe_contact_reward + self.task_complete_reward + self.force_reward)
        # )
        # self.reward_normalization_factor = horizon / (
        #     horizon * (self.wipe_contact_reward + self.force_reward)
        # )

        # Set task-specific parameters
        self.single_object_mode = single_object_mode
        self.obj_to_id = {"Flat_top": 0, "Incline": 1}
        self.obj_id_to_sensors = {}  # Maps obj id to sensor names for that obj
        if obj_type is not None:
            assert obj_type in self.obj_to_id.keys(), "invalid @obj_type argument - choose one of {}".format(
                list(self.obj_to_id.keys())
            )
            self.obj_id = self.obj_to_id[obj_type]  # use for convenient indexing
        self.obj_to_use = None

        # ee resets
        self.ee_force_bias = np.array([0, 0, -6.955])  # weight of eef
        self.ee_torque_bias = np.zeros(3)

        # set other wipe-specific attributes
        self.wiped_markers = []
        self.metadata = []
        self.spec = "spec"

        # object placement initializer
        self.placement_initializer = placement_initializer

        # whether to include and use ground-truth object states
        # self.use_object_obs = self.task_config["use_object_obs"]
        self.use_object_obs = True

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, reward_scaled_min=-1, reward_scaled_max=1):
        """
        Calculate the reward for the wiping task.
        The following components are considered:
        - Force penalty (penalty_force): |F_ee - F^target|
        - x velocity penalty (penalty_xvel): |v_x - v_x^target|
        - y velocity penalty (penalty_yvel): |v_y - v_y^target|
        - x distance penalty (penalty_xdist): |p_x - p_x^target|
        - reward for done (reward_done)
        - reward contact (reward_contact)

        All components are in the [0,1] range.

        The final reward is given by:
                reward =   self.reward_calc_c_force * penalty_force \
                 + self.reward_calc_c_xvel * penalty_xvel \
                 + self.reward_calc_c_xdist * penalty_xdist \
                 + self.reward_calc_c_yvel * penalty_yvel \
                 + self.reward_calc_c_contact * reward_contact \
                 + self.reward_calc_c_done * reward_done

        And scaled to [-1,1].
        :return: Reward value [-1,1]
        """
        # Safety Feature: Negative penalty from collisions of the arm with the table
        # ToDo: This check returns ALWAYS false. No check against Robot performed.
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                print("penalizing contact")
                reward = self.arm_limit_collision_penalty
            self.collisions = 1
            return reward

        # Safety Feature: Negative penalty for robot at joint limits
        # ToDo: This check returns ALWAYS false. No check against Robot performed.
        if self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
                print("penalizing q_limit")
            self.joint_limits = 1
            return reward

        # If the arm is not colliding or in joint limits, we check if we are wiping
        # (we don't want to reward wiping if there are unsafe situations)
        # Compute penalty/reward contributions
        # Every penalty/reward should be normalized to [0,1]

        # Compute force penalty (penalty_force)
        self.total_force_ee = np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)
        self.penalty_force = _compute_penalty(self.total_force_ee, self.target_force,
                                              self.reward_calc_lower_limit_force,
                                              self.reward_calc_upper_limit_force)

        # Compute x velocity penalty (penalty_xvel)
        #xvel_ee = self.sim.data.get_site_xvelp(self.robots[0].eef_site_id)[0]
        #xvel_ee = self.sim.data.get_site_xvelp('gripper0_ee_x')[0]
        xvel_ee = self.robots[0]._hand_vel[0]
        self.penalty_xvel = _compute_penalty(xvel_ee, self.target_xvel, self.reward_calc_lower_limit_xvel,
                                             self.reward_calc_upper_limit_xvel)

        # Compute x distance penalty (penalty_xdist)
        goal_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(self.objs[0].sites[7])]
        xdist_ee = np.linalg.norm(self._eef_xpos[0] - goal_pos[0])
        self.penalty_xdist = _compute_penalty(xdist_ee, self.target_xdist, self.reward_calc_lower_limit_xdist,
                                              self.reward_calc_upper_limit_xdist)

        # Compute y velocity penalty (penalty_yvel)
        yvel_ee = self.robots[0]._hand_vel[1]  # ToDo: Is this the y velocity?
        self.penalty_yvel = _compute_penalty(yvel_ee, self.target_yvel, self.reward_calc_lower_limit_yvel,
                                             self.reward_calc_upper_limit_yvel)

        # Compute reward contact penalty (reward_contact)
        self.reward_contact = 0
        if self.total_force_ee > self.reward_calc_min_force_for_contact:
            # ToDo: To adhere to the scheme of rewards in [0,1] this setting is not necessary?
            self.reward_contact = 1  # self.wipe_contact_reward

        # Compute reward for done penalty (reward_done)
        self.reward_done = 0
        # check if all markers are wiped and give final reward if so
        self._update_wiped_markers()  # ToDo: What does this function do? Copied from Aditya
        if self.wiped_markers:
            if self.wiped_markers[-1] == self.objs[0].sites[-3]:
                self.reward_done = self.task_config.reward_done  # should this be the number of episode steps?

        reward = self.reward_calc_c_force * self.penalty_force \
                 + self.reward_calc_c_xvel * self.penalty_xvel \
                 + self.reward_calc_c_xdist * self.penalty_xdist \
                 + self.reward_calc_c_yvel * self.penalty_yvel \
                 + self.reward_calc_c_contact * self.reward_contact \
                 + self.reward_calc_c_done * self.reward_done

        reward_min = self.reward_calc_c_force + self.reward_calc_c_xvel + self.reward_calc_c_xdist + self.reward_calc_c_yvel
        reward_max = self.reward_calc_c_contact

        # Scale the reward to desired range
        reward_scaled = (reward - reward_min) / (reward_max - reward_min) * np.abs(
            reward_scaled_max - reward_scaled_min) + reward_scaled_min
        return reward_scaled

    def _update_wiped_markers(self):
        active_markers = []

        # Only go into this computation if there are contact points
        if self.total_force_ee >= 0 and self.sim.data.ncon != 0:

            # Check each marker that is still active
            for marker in self.objs[0].sites[:-1]:

                # Current marker 3D location in world frame
                marker_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(marker)])

                end_face_centroid = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                v = marker_pos - end_face_centroid

                v_dist = np.linalg.norm(v)

                if v_dist < 0.02:
                    active_markers.append(marker)

        # Obtain the list of currently active (wiped) markers that where not wiped before
        # These are the markers we are wiping at this step
        lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
        new_active_markers = np.array(active_markers)[lall]
        # Loop through all new markers we are wiping at this step
        for new_active_marker in new_active_markers:
            # Add this marker the wiped list
            new_active_marker_geom_id = self.sim.model.site_name2id(new_active_marker)
            # Make this marker transparent since we wiped it (alpha = 0)
            self.sim.model.site_rgba[new_active_marker_geom_id][3] = 0
            self.wiped_markers.append(new_active_marker)

            # Add reward if we're using the dense reward
            # self.unit_wipe = self.unit_wiped_reward  # logging_purposes

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms
        delta_height = np.clip(np.random.normal(self.table_height, self.table_height_std), a_min=-0.3,
                               a_max=0)  # sample variation in height
        print(f'delta_height: {delta_height}')
        self.table_offset = np.array(self.task_config["table_offset"]) + np.array((0, 0, delta_height))
        print(f'resetting_table_height, new_table_height:{self.table_offset}')
        mujoco_arena = PolishingArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std
        )
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # define objects
        self.objs = []
        obj_names = ("Flat_top", "Incline")

        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        # if self.placement_initializer is None:       
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for obj_name, default_y_range in zip(obj_names, ([0, 0], [0, 0])):
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{obj_name}Sampler",
                    x_range=[0, 0],
                    y_range=default_y_range,
                    rotation=None,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.0,
                )
            )
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (obj_cls, obj_name) in enumerate(
                zip(
                    (Flat_top, Incline),
                    obj_names,
                )
        ):
            obj = obj_cls(name=obj_name)
            self.objs.append(obj)

            # Add this obj to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add objs to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{obj_name}Sampler", mujoco_objects=obj)
            else:
                # This is assumed to be a flat sampler, so we just add all objs to this sampler
                self.placement_initializer.add_objects(obj)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objs,
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        sensors = []
        names = []
        # object information in the observation

        '''
        Adding force sensor reading in the observation
        '''
        if self.use_object_obs:
            if self.use_force_obs:
                @sensor(modality=modality)
                def force_reading(obs_cache):
                    obs_cache["eef_force"] = self.total_force_ee
                    return self.total_force_ee if "eef_force" in obs_cache else np.zeros(1)

                sensors += [force_reading]
                names += ["eef_force"]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_marker_sensors(self, i, marker, modality="object"):
        """
        Helper function to create sensors for a given marker. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            i (int): ID number corresponding to the marker
            marker (MujocoObject): Marker to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given marker
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def marker_pos(obs_cache):
            # return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])
            return np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.objs[0].sites[i])])

        @sensor(modality=modality)
        def marker_wiped(obs_cache):
            return [0, 1][marker in self.wiped_markers]

        sensors = [marker_pos, marker_wiped]
        names = [f"marker{i}_pos", f"marker{i}_wiped"]

        if self.use_robot_obs:
            # also use ego-centric obs
            @sensor(modality=modality)
            def gripper_to_marker(obs_cache):
                return (
                    obs_cache[f"marker{i}_pos"] - obs_cache[f"{pf}eef_pos"]
                    if f"marker{i}_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors.append(gripper_to_marker)
            names.append(f"gripper_to_marker{i}")

        return sensors, names

    def _reset_internal(self):
        super()._reset_internal()

        # inherited class should reset positions of objects (only if we're not using a deterministic reset)
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # print(obj_pos, obj)
                try:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                except:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    # for marker in self.objs[0].sites:
                    #     self.sim.model.site_rgba[self.sim.model.site_name2id(marker)][3]=1

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objs}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(obj_names))
            for obj_type, i in self.obj_to_id.items():
                if obj_type.lower() in self.obj_to_use.lower():
                    self.obj_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objs[self.obj_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        # Reset all internal vars for this wipe task
        self.timestep = 0
        self.wiped_markers = []
        self.collisions = 0
        self.f_excess = 0

        # ee resets - bias at initial state
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

    def _check_success(self):
        """
        Checks if Task succeeds (all dirt wiped).

        Returns:
            bool: True if completed task
        """

        # return True if len(self.wiped_markers) == self.num_markers else False
        return True if self.reward_done > 0  else False

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision
            - Task completion (wiping succeeded)
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        if self.robots[0]._hand_vel[1] < 0:
            if self.print_results:
                print(40 * "-" + " NEGATIVE VELOCITY " + 40 * "-")
                terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            if self.print_results:
                print(40 * "-" + " COLLIDED " + 40 * "-")
            terminated = True

        # Prematurely terminate if task is success
        if self._check_success():
            if self.print_results:
                print(40 * "+" + " FINISHED WIPING " + 40 * "+")
            terminated = True

        # Prematurely terminate if contacting the table with the arm
        if self.robots[0].check_q_limits():
            if self.print_results:
                print(40 * "-" + " JOINT LIMIT " + 40 * "-")
            terminated = True

        return terminated

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        # Update force bias
        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        if self.get_info:
            info["add_vals"] = ["nwipedmarkers", "colls", "lims", "percent_viapoints_", "f_excess", "force",
                                "deviation", "wiped_via_point", "table_height"]
            info["nwipedmarkers"] = len(self.wiped_markers)
            info["colls"] = self.collisions
            info["lims"] = self.joint_limits
            info["percent_viapoints_"] = len(self.wiped_markers) / self.num_markers
            info["f_excess"] = self.f_excess
            info["force"] = np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)
            #info["deviation"] = self.x_dist
            info["wiped_via_point"] = self.wiped_markers
            info["table_height"] = self.table_offset[-1]

            info["reward_done"]=self.reward_done
            info["reward_contact"]=self.reward_contact
            info["penalty_yvel"]=self.penalty_yvel
            info["penalty_xdist"]=self.penalty_xdist
            info["penalty_force"]=self.penalty_force
            info["penalty_xvel"]=self.penalty_xvel
        # allow episode to finish early if allowed
        if self.early_terminations:
            done = done or self._check_terminated()

        return reward, done, info

    def _get_wipe_information(self):
        """Returns set of wiping information"""

        mean_pos_to_things_to_wipe = np.zeros(3)
        wipe_centroid = np.zeros(3)
        marker_positions = []
        num_non_wiped_markers = 0
        if len(self.wiped_markers) < self.num_markers:
            for marker in self.objs[0].sites[:-1]:
                # for marker in self.model.mujoco_arena.markers:
                if marker not in self.wiped_markers:
                    marker_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(marker)])
                    wipe_centroid += marker_pos
                    marker_positions.append(marker_pos)
                    num_non_wiped_markers += 1
            wipe_distance = np.sum(
                [np.linalg.norm(y - marker_positions[x - 1]) for x, y in enumerate(marker_positions)][1:])
            wipe_centroid /= max(1, num_non_wiped_markers)
            total_wipe_distance = np.linalg.norm(self._eef_xpos - marker_positions[0]) + wipe_distance
            mean_pos_to_things_to_wipe = wipe_centroid - self._eef_xpos
        # Radius of circle from centroid capturing all remaining wiping markers
        max_radius = 0
        if num_non_wiped_markers > 0:
            max_radius = np.max(np.linalg.norm(np.array(marker_positions) - wipe_centroid, axis=1))
        # Return all values
        return max_radius, wipe_centroid, mean_pos_to_things_to_wipe, total_wipe_distance

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        # print(np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold)
        return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold
        # return True

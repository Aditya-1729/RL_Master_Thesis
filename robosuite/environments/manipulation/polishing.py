import random
from collections import OrderedDict
import multiprocessing
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import PolishingArena
# from robosuite.models.objects import RoundobjObject, SquareobjObject
from robosuite.models.objects import Flat_top, Incline
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
DEFAULT_WIPE_CONFIG = {
    # settings for reward
    "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
    "wipe_contact_reward": 0.5,  # reward for contacting something with the wiping tool  #0.1
    "unit_wiped_reward": 50.0,  # reward per peg wiped
    "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.5,  # penalty for each step that the force is over the safety threshold       #0.05
    "distance_multiplier": 10.0,  # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
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
    "target_force":3.5,
    # misc settings
    "print_results": False,  # Whether to print results or not
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": False,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,  # Whether we allow for early terminations or not
    "use_condensed_obj_obs": False,  # Whether to use condensed object observation representation (only applicable if obj obs is active)
    "no_contact_penalty":0,
    "force_multiplier":0.5,
    "reward_mode": 0,

    
}


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
        initialization_noise=None, #"default"
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        single_object_mode=2, #mode single to randomly select one of the two objects on reset
        obj_type="Flat_top",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1500,
        _max_episode_steps = 100,
        ignore_done=False,
        hard_reset=False,
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
        self.task_config = task_config if task_config is not None else DEFAULT_WIPE_CONFIG

        #adding this because the wrapper threw an error for getting attribute
        # self._max_episode_steps = _max_episode_steps

        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config["arm_limit_collision_penalty"]
        self.wipe_contact_reward = self.task_config["wipe_contact_reward"]
        self.unit_wiped_reward = self.task_config["unit_wiped_reward"]
        self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        self.excess_force_penalty_mul = self.task_config["excess_force_penalty_mul"]
        self.distance_multiplier = self.task_config["distance_multiplier"]
        self.distance_th_multiplier = self.task_config["distance_th_multiplier"]
        self.force_multiplier = self.task_config["force_multiplier"]
        self.target_force = self.task_config["target_force"]
        self.general_penalty = self.task_config["general_penalty"]

        self.dist_th = self.task_config["dist_th"]
        #Reward for maintaining force in the right window
        self.reward_mode = self.task_config["reward_mode"]
        self.force_reward = self.force_multiplier*self.target_force

        #vel_threshold
        self.min_vel = self.task_config["min_vel"]
        self.max_vel = self.task_config["max_vel"]

        #position tracking 
        self.position_track_multiplier = self.task_config["position_track_multiplier"]
        # Final reward computation
        # So that is better to finish that to stay touching the table for 100 steps
        # The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
        # self.task_complete_reward = self.unit_wiped_reward * (self.wipe_contact_reward + 0.5)
        self.task_complete_reward =self.task_config["task_complete_reward"]
        # Verify that the distance multiplier is not greater than the task complete reward
        assert (
            self.task_complete_reward > self.distance_multiplier
        ), "Distance multiplier cannot be greater than task complete reward!"

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
        self.num_markers = 8 #len(self.objs[0].sites[:-1])      #it wont work like this

        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold = self.task_config["pressure_threshold"]
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]
        self.f_cap = self.task_config["f_safe"]


        # misc settings
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
        self.reward_normalization_factor = horizon / (
             horizon * (self.wipe_contact_reward + self.force_reward)
        )

        # Set task-specific parameters
        self.single_object_mode = single_object_mode
        self.obj_to_id = {"Flat_top": 0, "Incline":1}
        self.obj_id_to_sensors = {}  # Maps obj id to sensor names for that obj
        if obj_type is not None:
            assert obj_type in self.obj_to_id.keys(), "invalid @obj_type argument - choose one of {}".format(
                list(self.obj_to_id.keys())
            )
            self.obj_id = self.obj_to_id[obj_type]  # use for convenient indexing
        self.obj_to_use = None


        # ee resets
        self.ee_force_bias = np.array([0,0,-6.955]) #weight of eef
        self.ee_torque_bias = np.zeros(3)

        # set other wipe-specific attributes
        self.wiped_markers = []
        self.metadata = []
        self.spec = "spec"

        # object placement initializer
        self.placement_initializer = placement_initializer


        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

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

    def reward(self, action=None):

        self.force_in_window_penalty=0
        self.force_penalty =0
        self.task_completion_r = 0
        self.unit_wipe=0
        self.wipe_contact_r=0
        self.low_force_penalty = 0
        self.collisions = 0
        self.joint_limits=0
        self.f_excess = 0

        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of self.unit_wiped_reward is provided per single dirt (peg) wiped during this step
            - a discrete reward of self.task_complete_reward is provided if all dirt is wiped

        Note that if the arm is either colliding or near its joint limit, a reward of 0 will be automatically given

        Un-normalized summed components if using reward shaping (individual components can be set to 0:

            - Reaching: in [0, self.distance_multiplier], proportional to distance between wiper and centroid of dirt
              and zero if the table has been fully wiped clean of all the dirt
            - Table Contact: in {0, self.wipe_contact_reward}, non-zero if wiper is in contact with table
            - Wiping: in {0, self.unit_wiped_reward}, non-zero for each dirt (peg) wiped during this step
            - Cleaned: in {0, self.task_complete_reward}, non-zero if no dirt remains on the table
            - Collision / Joint Limit Penalty: in {self.arm_limit_collision_penalty, 0}, nonzero if robot arm
              is colliding with an object
              - Note that if this value is nonzero, no other reward components can be added
            - Large Force Penalty: in [-inf, 0], scaled by wiper force and directly proportional to
              self.excess_force_penalty_mul if the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0], scaled by estimated wiper acceleration and directly
              proportional to self.ee_accel_penalty

        Note that the final per-step reward is normalized given the theoretical best episode return and then scaled:
        reward_scale * (horizon /
        (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0
        
        # total_force_ee = np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3]))
        total_force_ee = np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)
        goal_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(self.objs[0].sites[7])]

        self.x_dist = np.linalg.norm(self._eef_xpos[0] - goal_pos[0])
        # Neg Reward from collisions of the arm with the table
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                print("penalizing contact")
                reward = self.arm_limit_collision_penalty
            self.collisions = 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
                print("penalizing q_limit")
            self.joint_limits = 1
        else:
            # If the arm is not colliding or in joint limits, we check if we are wiping
            # (we don't want to reward wiping if there are unsafe situations)
            active_markers = []

            # Current 3D location of the corners of the wiping tool in world frame
            c_geoms = self.robots[0].gripper.important_geoms["corners"]
            corner1_id = self.sim.model.geom_name2id(c_geoms[0])
            corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
            corner2_id = self.sim.model.geom_name2id(c_geoms[1])
            corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
            corner3_id = self.sim.model.geom_name2id(c_geoms[2])
            corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
            corner4_id = self.sim.model.geom_name2id(c_geoms[3])
            corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])

            # Unit vectors on my plane
            v1 = corner1_pos - corner2_pos
            v1 /= np.linalg.norm(v1)
            v2 = corner4_pos - corner2_pos
            v2 /= np.linalg.norm(v2)


            # Corners of the tool in the coordinate frame of the plane
            t1 = np.array([np.dot(corner1_pos - corner2_pos, v1), np.dot(corner1_pos - corner2_pos, v2)])
            t2 = np.array([np.dot(corner2_pos - corner2_pos, v1), np.dot(corner2_pos - corner2_pos, v2)])
            t3 = np.array([np.dot(corner3_pos - corner2_pos, v1), np.dot(corner3_pos - corner2_pos, v2)])
            t4 = np.array([np.dot(corner4_pos - corner2_pos, v1), np.dot(corner4_pos - corner2_pos, v2)])

            pp = [t1, t2, t4, t3]

            # Normal of the plane defined by v1 and v2
            n = np.cross(v2,v1) #changed
            n /=np.linalg.norm(n)
            
            def isLeft(P0, P1, P2):
                return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

            def PointInRectangle(X, Y, Z, W, P):
                return isLeft(X, Y, P) < 0 and isLeft(Y, Z, P) < 0 and isLeft(Z, W, P) < 0 and isLeft(W, X, P) < 0

            # Only go into this computation if there are contact points
            if self.sim.data.ncon != 0:

                # Check each marker that is still active
                for marker in self.objs[0].sites[:-1]:

                    # Current marker 3D location in world frame
                    marker_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(marker)])
                    
                    # We use the second tool corner as point on the plane and define the vector connecting
                    # the marker position to that point
                    end_face_centroid = (corner1_pos+corner2_pos+corner3_pos+corner4_pos)/4
                    v = marker_pos - end_face_centroid
                    # print(f'centroid:{end_face_centroid}')
                    # v = marker_pos - corner2_pos
                    v_dist = np.linalg.norm(v)
                    # Shortest distance between the center of the marker and the plane
                    dist = np.dot(v, n)
                    # print("v: {}, marker: {}, dist: {}, ".format(v_dist, marker, dist))

                    # Projection of the center of the marker onto the plane
                    projected_point = np.array(marker_pos) - dist * n

                    # Positive distances means the center of the marker is over the plane
                    # The plane is aligned with the bottom of the wiper and pointing up, so the marker would be over it
                    # if dist < 0.0:
                        # Distance smaller than this threshold means we are close to the plane on the upper part
                    if v_dist < 0.02:
                        # Write touching points and projected point in coordinates of the plane
                        pp_2 = np.array(
                            [np.dot(projected_point - corner2_pos, v1), np.dot(projected_point - corner2_pos, v2)]
                        )
                        # Check if marker is within the tool center:
                        if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
                            
                            active_markers.append(marker)


            # Obtain the list of currently active (wiped) markers that where not wiped before
            # These are the markers we are wiping at this step
            lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
            new_active_markers = np.array(active_markers)[lall]
            # print("active_markers: {}, new_active_markers: {}".format(active_markers, new_active_markers) )
            # Loop through all new markers we are wiping at this step
            for new_active_marker in new_active_markers:
                # Grab relevant marker id info
                new_active_marker_geom_id = self.sim.model.site_name2id(new_active_marker)
                # Make this marker transparent since we wiped it (alpha = 0)
                self.sim.model.site_rgba[new_active_marker_geom_id][3] = 0
                # Add this marker the wiped list
                self.wiped_markers.append(new_active_marker)
                # Add reward if we're using the dense reward
                if self.reward_shaping:
                    
                    # reward += self.unit_wiped_reward #commented out the reward component
                    
                    self.unit_wipe = self.unit_wiped_reward #logging_purposes

            # Additional reward components if using dense rewards
            if self.reward_shaping and self.reward_mode==2:
                self.force_in_window_mul = self.task_config["force_in_window_mul"]
                self.low_force_penalty_mul = self.task_config["low_force_penalty_mul"]
                self.safe_force_low = self.task_config["safe_force_low"]
                self.safe_force_high = self.task_config["safe_force_high"]


                self.reward_normalization_factor = 1
                
                #set the cost range for eef_force
                self.max_force_cost = 1/self.reward_normalization_factor + self.wipe_contact_reward
                self.min_force_cost = 1/self.reward_normalization_factor + self.wipe_contact_reward
                
                
                # Reward for keeping contact
                if self.sim.data.ncon > 1 and self.min_vel<self.robots[0]._hand_vel[1]<self.max_vel:
                    # print("contact")
                    self.force_in_window_penalty = self.force_in_window_mul * np.square(total_force_ee - self.target_force)
                    # print(f"before_clip{self.force_penalty}")
                    # self.force_in_window_penalty = np.clip(self.force_in_window_penalty, 0, self.max_force_cost)
                    # print(f"after_clip{self.force_penalty}")
                    # reward = reward - self.force_in_window_penalty # + self.wipe_contact_reward
                    reward += self.wipe_contact_reward
                    self.wipe_contact_r = self.wipe_contact_reward

                # Penalty for excessive force with the end-effector
                    if total_force_ee >= self.safe_force_high:
                        self.force_penalty = self.excess_force_penalty_mul * np.square(total_force_ee - self.safe_force_high)
                        # print(f"before_clip{self.force_penalty}")
                        self.force_penalty = np.clip(self.force_penalty+self.force_in_window_penalty, a_min=0, a_max=self.max_force_cost)
                        # print(f"after_clip{self.force_penalty}")
                        reward = reward - self.force_penalty # + self.wipe_contact_reward
                        if total_force_ee>=self.f_cap:                    
                            self.f_excess = 1


                # Reward for pressing into table
                # TODO: Need to include this computation somehow in the scaled reward computation
                    elif total_force_ee > self.contact_threshold and total_force_ee<=self.safe_force_low:
                        
                        self.low_force_penalty = self.low_force_penalty_mul*(np.square(total_force_ee - self.safe_force_low)) #+self.wipe_contact_reward 
                        self.low_force_penalty = np.clip(self.low_force_penalty+self.force_in_window_penalty, a_min=0, a_max=self.min_force_cost)
                        reward -= self.low_force_penalty

                else:
                    self.wipe_contact_r=0
                    reward=self.general_penalty

                # logging_purposes
                self.reward_wop = reward
                #progress reward
                total_distance=np.linalg.norm(self._eef_xpos - goal_pos)
                self.x_dist = np.linalg.norm(self._eef_xpos[0] - goal_pos[0])
                self.total_dist_reward = self.distance_multiplier*np.tanh(total_distance)
                reward-=self.total_dist_reward
                reward-= self.position_track_multiplier*self.x_dist

                if self.wiped_markers:
                    if self.wiped_markers[-1] == self.objs[0].sites[-3]:
                        # print("completed_task")
                        self.task_completion_r = self.task_complete_reward
                        reward += self.task_complete_reward

                # Penalize large accelerations
                reward -= self.ee_accel_penalty * np.mean(abs(self.robots[0].recent_ee_acc.current))


            if self.reward_shaping and self.reward_mode==3:
                self.force_in_window_mul = self.task_config["force_in_window_mul"]
                self.low_force_penalty_mul = self.task_config["low_force_penalty_mul"]
                self.safe_force_low = self.task_config["safe_force_low"]
                self.safe_force_high = self.task_config["safe_force_high"]


                self.reward_normalization_factor = 1
                
                #set the cost range for eef_force
                self.max_force_cost = 1/self.reward_normalization_factor + self.wipe_contact_reward
                self.min_force_cost = 1/self.reward_normalization_factor + self.wipe_contact_reward
                
                
                # Reward for keeping contact
                if self.sim.data.ncon > 1 and self.min_vel<self.robots[0]._hand_vel[1]<self.max_vel:
                    # print("contact")
                    self.force_in_window_penalty = self.force_in_window_mul * np.square(total_force_ee - self.target_force)
                    # print(f"before_clip{self.force_penalty}")
                    # self.force_in_window_penalty = np.clip(self.force_in_window_penalty, 0, self.max_force_cost)
                    # print(f"after_clip{self.force_penalty}")
                    reward = reward - self.force_in_window_penalty # + self.wipe_contact_reward
                    reward += self.wipe_contact_reward
                    self.wipe_contact_r = self.wipe_contact_reward

                # Penalty for excessive force with the end-effector
                    if total_force_ee >= self.safe_force_high:
                        self.force_penalty = self.excess_force_penalty_mul * np.square(total_force_ee - self.safe_force_high)
                        # print(f"before_clip{self.force_penalty}")
                        self.force_penalty = np.clip(self.force_penalty, 0, self.max_force_cost)
                        # print(f"after_clip{self.force_penalty}")
                        reward = reward - self.force_penalty # + self.wipe_contact_reward
                        self.f_excess += 1


                # Reward for pressing into table
                # TODO: Need to include this computation somehow in the scaled reward computation
                    elif total_force_ee > self.contact_threshold and total_force_ee<=self.safe_force_low:
                        
                        self.low_force_penalty = self.low_force_penalty_mul*(np.square(total_force_ee - self.safe_force_low)) #+self.wipe_contact_reward 
                        self.low_force_penalty = np.clip(self.low_force_penalty, 0, self.min_force_cost)
                        reward -= self.low_force_penalty

                else:
                    self.wipe_contact_r=0
                    reward=-1

                # logging_purposes
                self.reward_wop = reward
                #progress reward
                goal_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(self.objs[0].sites[7])]
                total_distance=np.linalg.norm(self._eef_xpos - goal_pos)
                self.total_dist_reward = self.distance_multiplier*np.tanh(total_distance)
                reward-=self.total_dist_reward

                if self.wiped_markers:
                    if self.wiped_markers[-1] == self.objs[0].sites[-3]:
                        # print("completed_task")
                        self.task_completion_r = self.task_complete_reward
                        reward += self.task_complete_reward

                # Penalize large accelerations
                reward -= self.ee_accel_penalty * np.mean(abs(self.robots[0].recent_ee_acc.current))
            

            if self.reward_shaping and self.reward_mode==4:
                self.force_in_window_mul = self.task_config["force_in_window_mul"]
                self.low_force_penalty_mul = self.task_config["low_force_penalty_mul"]
                self.safe_force_low = self.task_config["safe_force_low"]
                self.safe_force_high = self.task_config["safe_force_high"]


                self.reward_normalization_factor = 1
                
                #set the cost range for eef_force
                self.max_force_cost = 1/self.reward_normalization_factor + self.wipe_contact_reward
                self.min_force_cost = 1/self.reward_normalization_factor + self.wipe_contact_reward
                
                
                # Reward for keeping contact
                if self.sim.data.ncon > 1 and self.min_vel<self.robots[0]._hand_vel[1]<self.max_vel:
                    # print("contact")
                    self.force_in_window_penalty = self.force_in_window_mul * np.square(total_force_ee - self.target_force)
                    # print(f"before_clip{self.force_penalty}")
                    # self.force_in_window_penalty = np.clip(self.force_in_window_penalty, 0, self.max_force_cost)
                    # print(f"after_clip{self.force_penalty}")
                    reward = reward - self.force_in_window_penalty # + self.wipe_contact_reward
                    reward += self.wipe_contact_reward
                    self.wipe_contact_r = self.wipe_contact_reward

                # Penalty for excessive force with the end-effector
                    if total_force_ee >= self.safe_force_high:
                        self.force_penalty = self.excess_force_penalty_mul * np.square(total_force_ee - self.safe_force_high)
                        # print(f"before_clip{self.force_penalty}")
                        self.force_penalty = np.clip(self.force_penalty, 0, self.max_force_cost)
                        # print(f"after_clip{self.force_penalty}")
                        reward = reward - self.force_penalty # + self.wipe_contact_reward
                        self.f_excess += 1


                # Reward for pressing into table
                # TODO: Need to include this computation somehow in the scaled reward computation
                    elif total_force_ee > self.contact_threshold and total_force_ee<=self.safe_force_low:
                        
                        self.low_force_penalty = self.low_force_penalty_mul*(np.square(total_force_ee - self.safe_force_low)) #+self.wipe_contact_reward 
                        self.low_force_penalty = np.clip(self.low_force_penalty, 0, self.min_force_cost)
                        reward -= self.low_force_penalty

                else:
                    self.wipe_contact_r=0
                    reward=self.general_penalty

                # logging_purposes
                self.reward_wop = reward
                #progress reward
                goal_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(self.objs[0].sites[7])]
                total_distance=np.linalg.norm(self._eef_xpos - goal_pos)
                self.total_dist_reward = self.distance_multiplier*np.tanh(total_distance)
                reward-=self.total_dist_reward


                # Penalize large accelerations
                reward -= self.ee_accel_penalty * np.mean(abs(self.robots[0].recent_ee_acc.current))
            
                if len(self.wiped_markers) == self.num_markers:
                    # print("completed_task")
                    self.task_completion_r = self.task_complete_reward
                    reward += self.task_complete_reward


        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}"
                "wiped markers: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    ws=len(self.wiped_markers),
                    sc=self.collisions,
                    fe=self.f_excess,
                )
            )
            print(string_to_print)

        # If we're scaling our reward, we normalize the per-step rewards given the theoretical best episode return
        # This is equivalent to scaling the reward by:
        #   reward_scale * (horizon /
        #       (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))

        #commented out 
        
        if self.reward_scale:
            self.un_normalized_reward = reward
            reward *= self.reward_scale * self.reward_normalization_factor
        
        return reward

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
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for obj_name, default_y_range in zip(obj_names, ([0, 0], [0,0])):
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
        # self.placement_initializer.reset()

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
        if self.use_object_obs:

            if self.use_condensed_obj_obs:
                # use implicit representation of wiping objects
                @sensor(modality=modality)
                def wipe_radius(obs_cache):
                    wipe_rad, wipe_cent, _,_ = self._get_wipe_information()
                    obs_cache["wipe_centroid"] = wipe_cent
                    return wipe_rad

                @sensor(modality=modality)
                def wipe_centroid(obs_cache):
                    return obs_cache["wipe_centroid"] if "wipe_centroid" in obs_cache else np.zeros(3)

                @sensor(modality=modality)
                def proportion_wiped(obs_cache):
                    return len(self.wiped_markers) / self.num_markers

                sensors += [proportion_wiped, wipe_radius, wipe_centroid]
                names += ["proportion_wiped", "wipe_radius", "wipe_centroid"]

                if self.use_robot_obs:
                    # also use ego-centric obs
                    @sensor(modality=modality)
                    def gripper_to_wipe_centroid(obs_cache):
                        return (
                            obs_cache["wipe_centroid"] - obs_cache[f"{pf}eef_pos"]
                            if "wipe_centroid" in obs_cache and f"{pf}eef_pos" in obs_cache
                            else np.zeros(3)
                        )

                    sensors.append(gripper_to_wipe_centroid)
                    names.append("gripper_to_wipe_centroid")
                
            else:
            
                    # use explicit representation of wiping objects
                for i, marker in enumerate(self.objs[0].sites[:-1]):
                    marker_sensors, marker_sensor_names = self._create_marker_sensors(i, marker, modality)
                    sensors += marker_sensors
                    names += marker_sensor_names
            
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
        return True if self.task_completion_r else False


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

        if self.robots[0]._hand_vel[1] <0:
            if self.print_results:
                print(40 * "-" + " NEGATIVE VELOCITY " + 40 * "-")
                terminated =True

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
            info["add_vals"] = ["nwipedmarkers", "colls", "lims", "percent_viapoints_", "f_excess", "force", "deviation", "wiped_via_point"]
            info["nwipedmarkers"] = len(self.wiped_markers)
            info["colls"] = self.collisions
            info["lims"] = self.joint_limits
            info["percent_viapoints_"] = len(self.wiped_markers) / self.num_markers
            info["f_excess"] = self.f_excess
            info["force"] = np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)
            info["deviation"] = self.x_dist
            info["wiped_via_point"] = self.wiped_markers
    

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
            wipe_distance = np.sum([np.linalg.norm(y - marker_positions[x - 1]) for x, y in enumerate(marker_positions)][1:])
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
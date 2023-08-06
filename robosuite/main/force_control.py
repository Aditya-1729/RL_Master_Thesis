# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
# wandb.login()
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import imageio


if __name__ == "__main__":
    use_wandb=False

    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    if use_wandb==True:
        run = wandb.init(
        # Set the project where this run will be logged
        project="test_ros_osc",
        name='new_ee_new_robot_i=0.01_dist',
        config=cfg) 
        config = wandb.config
    
    env = suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller))
    #impedence mode and control delta
    # initialize the task
    # print("init_qpos: {}".format(env.robots[0].init_qpos))
    dist_th = cfg.task_config.dist_th
    # dist_th =0.02
    indent=cfg.task_config.indent
    clip = cfg.task_config.clip

    env.reset()
    # writer = imageio.get_writer("recording_nominal.mp4",  format='FFMPEG', fps=20)
    Return=0
    Return_unnormalized=0
    t=0
    t_contact=0
    # do visualization
    sites =env.objs[0].sites[:-1]
    # sites.reverse()
    sites = cycle(sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    # site_pos_1 = env.sim.data.site_xpos[env.sim.model.site_name2id(env.objs[0].sites[-2])]
    action = np.array((0.15,-0.2,1))
    while site!=env.objs[0].sites[-1]:
        a = site_pos
        # a=np.array((0.15,-0.2,1.5))
        t+=1
        # action[:2] = a[:2]
        # action[-1] = a[-1] - indent
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        # print(eef_pos)
        action[:2] = eef_pos[:2] + np.clip(a[:2]-eef_pos[:2], a_min=np.array([-clip, -clip]), a_max=np.array([clip, clip]))
        action[-1] = eef_pos[-1] + np.clip(a[-1] - indent - eef_pos[-1], a_min = -clip, a_max=clip) 
        # total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        total_force = np.linalg.norm(env.robots[0].ee_force - env.ee_force_bias)
        # total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))
        if total_force>1:
            t_contact+=1
        # if env.sim.data.ncon >=4:
        #     t_contact+=1desired_force
        # print(action)
        obs, reward, done, info = env.step(action)
        # print(obs)
        # frame = obs[cfg.env.specs.camera_names + "_image"]
        # writer.append_data(frame)
        env.render()

        eef_pos = obs["robot0_eef_pos"]
        # print(f"eef_pos:{eef_pos}")
        delta = eef_pos - site_pos

        dist = np.linalg.norm(delta)
        Return +=reward
        Return_unnormalized +=env.un_normalized_reward
        if use_wandb==True:
            metrics = { 'Wipe/reward': reward,
                            'Wipe/Return': Return,
                            'Wipe/force': total_force,
                            'Wipe/desired_wrench': np.linalg.norm(env.robots[0].controller.decoupled_wrench),
                            'Wipe/desired_force': np.linalg.norm(env.robots[0].controller.desired_force),
                            "Wipe/force_reward": env.force_in_window_reward,
                            "Wipe/task_complete_reward": env.task_completion_r,
                            "Wipe/force_penalty": env.force_penalty,
                            "Wipe/low_force_penalty":env.low_force_penalty,
                            "Wipe/distance_penalty":-env.total_dist_reward,
                            "Wipe/wipe_contact_reward": env.wipe_contact_r,
                            "Wipe/Return_unnormalized": Return_unnormalized,
                            "Wipe/unnormalized_reward": env.un_normalized_reward,
                            "Wipe/unit_wiped_reward": env.unit_wipe,
                            "Wipe/time_ratio":t_contact/t,
                            'Wipe/dist': dist,
                            'Wipe/ncontacts': env.sim.data.ncon,
                            'Wipe/vel_x': env.robots[0]._hand_vel[0],
                            'Wipe/vel_y': env.robots[0]._hand_vel[1],
                            'Wipe/vel_z': env.robots[0]._hand_vel[2]
                            }
            
            wandb.log({**metrics})

        
        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, force : {}, q: {}".format(site, dist, Return, t, total_force, env.sim.data.qpos))
            site = next(sites)
            # print(f"robot_joint_pos: {obs['robot0_joint_pos']}")
            # print(f"robot_joint_pos: {env.robots[0]._joint_positions}")
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if env.task_completion_r:
            print("final_q_pos:{}".format(env.sim.data.qpos)) #removed horizon limit
            # env.reset()
            break
    # wandb.finish()
# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
import os
wandb.login()
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
import hydra


if __name__ == "__main__":
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "20"
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    
    initialize_config_dir(version_base=None, config_dir=os.path.abspath(os.path.join(scriptDir,'config/')))
    cfg = compose(config_name="main")


    default_config = OmegaConf.to_container(cfg)
    # config = wandb.config
    wandb.init(config=default_config)
    config = wandb.config
    
    env = suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller))

    # set agent config to be zero
    dist_th = cfg.task_config.dist_th
    indent = config.indent
    clip = cfg.task_config.clip

    env.reset()
    init_qpos=[-0.20805232,0.69171682, -0.12398874, -2.03316517,0.19272576, 2.71429916, 0.30621991]
    env.robots[0].set_robot_joint_positions(init_qpos)
    env.robots[0].controller.update_initial_joints(init_qpos)
    Return=0
    t=0
    # do visualization
    sites = cycle(env.objs[0].sites[:-1])
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]

    action = np.empty(15,)
    while site!=env.objs[0].sites[-2]:
        a = site_pos
        t+=1
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        # action = np.empty(15,)
        action[:2] = [config.Kd_x, config.Kd_y]
        action[2] = config.Kd_z
        action[3:5] = [config.Kd_or_x, config.Kd_or_y] 
        action[5] = config.Kd_or_z
        action[6:8]= [config.Kp_x, config.Kp_y]
        action[8] = config.Kp_z
        action[9:11] = [config.Kp_or_x, config.Kp_or_y]
        action[11] = config.Kp_or_z
        action[12:14] = eef_pos[:2] + np.clip(a[:2]-eef_pos[:2], a_min=np.array([-clip, -clip]), a_max=np.array([clip, clip]))
        action[-1] = eef_pos[-1] + np.clip(a[-1] - indent - eef_pos[-1], a_min = -clip, a_max=clip)

        total_force = np.linalg.norm(env.robots[0].ee_force - env.ee_force_bias)
        total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))

        obs, reward, done, info = env.step(action)

        eef_pos = obs["robot0_eef_pos"]
        delta = eef_pos - site_pos

        dist = np.linalg.norm(delta)
        Return +=reward
        metrics = { 'Wipe/reward': reward,
                        'Wipe/Return': Return,
                        'Wipe/force': total_force,
                        'Wipe/desired_force': np.linalg.norm(env.robots[0].controller.desired_force),
                        # "Wipe/force_reward": env.force_in_window_reward,
                        "Wipe/task_complete_reward": env.task_completion_r,
                        "Wipe/wipe_contact_reward": env.wipe_contact_r,
                        "Wipe/unit_wiped_reward": env.unit_wipe,
                        "Wipe/moment":total_moment,
                        'Wipe/vel_y': env.robots[0]._hand_vel[1],
                        'Wipe/dist': dist,}
        
        wandb.log({**metrics})

        
        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, moment : {}".format(site, dist, Return, t, np.array(env.robots[0].recent_ee_forcetorques.current[:-3])))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if t==1500 or done:
            # env.reset()
            env.close()
            break

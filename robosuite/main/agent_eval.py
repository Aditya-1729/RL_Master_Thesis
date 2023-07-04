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
from robosuite.wrappers import RL_agent_2
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC


if __name__ == "__main__":
    
    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    run = wandb.init(
    # Set the project where this run will be logged
    project="agent_eval",
    name='RL_nc_2000_150_7',
    config=cfg)
    
    # Track hyperparameters and run metadata
    # print(OmegaConf.to_dict(cfg))
    # initialize(version_base=None, config_path="config/")
    # cfg = compose(config_name="main")
    # default_config = OmegaConf.to_yaml(cfg)
    # config = wandb.config
    env = RL_agent_2(suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller))
                    ,cfg)
                    
    
    #impedence mode and control delta

    # initialize the task

    dist_th = cfg.task_config.dist_th
    indent=cfg.task_config.indent
  
    env.reset()

    Return=0
    Return_unnormalized=0
    t=0
    t_contact=0

    model = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/saved_models/hpc_progress.zip")

    '''
    obs,_ = env.reset()
    for i in range(2000):
        eef_xpos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        action, _states = model.predict(obs, deterministic=True)
        # action = np.array((3,3,3,40,40,40))
        goal_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(env.objs[0].sites[8])]
        total_distance=np.linalg.norm(eef_xpos - goal_pos)
        total_dist_reward =  (
                - np.tanh(total_distance))
        print(f"penalty:{total_dist_reward}")
        print(f"gains: {action}")
        print(f"xpos:{env.sim.data.site_xpos[env.robots[0].eef_site_id][1]}")
        print(f"total_force: {np.linalg.norm(env.robots[0].ee_force)}")
        obs, _, _,_, info = env.step(action)
        env.render()
    '''


    # dist_th = 0.01
    # indent=0.003
    # writer = SummaryWriter('/home/adityapradhan/robosuite/robosuite/Runs/force_kp_{kp}_kd_{kd}_th_{th}_in_{ind}'.format(kp=controller_config['kp'], \
    # 
    #                         kd=controller_config['damping_ratio'], th=dist_th, ind=indent))

    '''
    env.reset()

    Return=0
    t=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    action = np.array((2.03,9.71,5.4,3,3,3, 280,21.8,51.2,40, 40, 40, 0.15,-0.2,0.945))
    while site!=env.objs[0].sites[-1]:
        a = site_pos
        t+=1
        # action[:2] = action[:2]
        # action[2] = config.Kd_z
        # action[3:6] = config.Kd_xy
        # action[6:8]= config.Kp_xy
        # action[8] = config.Kp_z
        # action[9:12] = config.Kp_xy
        action[12:14] = a[:2]
        action[-1] = a[-1] - indent
 
        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))

        obs, reward, done, info = env.step(action)
        env.render()
        print(total_force, env.sim.data.site_xpos[env.robots[0].eef_site_id])
        eef_pos = obs["robot0_eef_pos"]
        delta = eef_pos - site_pos

        dist = np.linalg.norm(delta)
        Return +=reward
        metrics = { 'Wipe/reward': reward,
                        'Wipe/Return': Return,
                        'Wipe/force': total_force,
                        "Wipe/force_reward": env.force_in_window_reward,
                        "Wipe/task_complete_reward": env.task_completion_r,
                        "Wipe/wipe_contact_reward": env.wipe_contact_r,
                        "Wipe/penalty": env.force_penalty,
                        "Wipe/unit_wiped_reward": env.unit_wipe,
                        "Wipe/moment":total_moment,
                        'Wipe/dist': dist,}
        
        wandb.log({**metrics})

        
        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, total_force : {}".format(site, dist, Return, t, total_force))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if t>2000 or env.task_completion_r:
            env.reset()
            break
    '''

    obs,_ = env.reset()

    Return=0
    t=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    while site!=env.objs[0].sites[-1]:
        action, _states = model.predict(obs, deterministic=True)
        # print(f"action:{action}")
        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))
        obs, reward, _, done, info = env.step(action)
        t+=1
        # env.render()
        # print(total_force, env.sim.data.site_xpos[env.robots[0].eef_site_id])
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        delta = eef_pos - site_pos

        dist = np.linalg.norm(delta)
        Return +=reward
        metrics = { 'Wipe/reward': reward,
                        'Wipe/kdy': action[1],
                        'Wipe/kpy': action[-2],
                        'Wipe/kpz': action[-3],
                        'Wipe/kpz': action[2],
                        'Wipe/Return': Return,
                        'Wipe/force': total_force,
                        "Wipe/force_reward": env.force_in_window_reward,
                        "Wipe/task_complete_reward": env.task_completion_r,
                        "Wipe/wipe_contact_reward": env.wipe_contact_r,
                        "Wipe/penalty": env.force_penalty,
                        "Wipe/unit_wiped_reward": env.unit_wipe,
                        "Wipe/moment":total_moment,
                        'Wipe/dist': dist,}
        wandb.log({**metrics})

        # print(Return)
        if dist < dist_th:
            # print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, total_force : {}".format(site, dist, Return, t, total_force))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if t>2000 or env.task_completion_r or done:
            env.reset()
            break
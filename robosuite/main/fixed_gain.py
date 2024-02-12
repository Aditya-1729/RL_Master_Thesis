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
# from robosuite.wrappers import RL_agent_2
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":
    
    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    if cfg.use_wandb:
        run = wandb.init(
        # Set the project where this run will be logged
        project='Default',
        name=f"fric_0.009 0.009 0.009 np spec",
        config=cfg) 
        config = wandb.config
    env = GymWrapper(
        suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller))
                    
    )
    #impedence mode and control delta

    # initialize the task

    dist_th = cfg.task_config.dist_th
    indent=cfg.task_config.indent

    env.reset()
    dist_th =0.020
    indent=0.01183
    position_limits = 0.020
    Return=0
    Return_unnormalized=0
    t=0
    t_contact=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]


    while True:
        a = site_pos
        t+=1

        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]

        action = np.empty(15)
        action[:12]=np.array((10,10,10,10,7,10,200,300,300,200,200,200))
        action[12:14] = eef_pos[:2] + np.clip(site_pos[:2]-eef_pos[:2], a_min=np.array([-position_limits, -position_limits]), a_max=np.array([position_limits, position_limits]))
        action[-1] = eef_pos[-1] + np.clip(site_pos[-1] - indent - eef_pos[-1], a_min = -position_limits, a_max=position_limits) 
        # total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        total_force = np.linalg.norm(env.robots[0].ee_force - env.ee_force_bias)
        # total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))
        obs, reward, done, _, info = env.step(action)
        # env.render()
        # print(obs)
        eef_pos = env._eef_xpos
        # print(f"eef_pos:{eef_pos}")
        delta = eef_pos - site_pos
        # print(done)
        dist = np.linalg.norm(delta)
        Return +=reward
        if cfg.use_wandb:
        # Return_unnormalized +=env.un_normalized_reward
            metrics = { 'Wipe/reward': reward,
                            'Wipe/Return': Return,
                            'Wipe/force': total_force,
                            # "Wipe/force_reward": env.force_in_window_reward,
                            "Wipe/task_complete_reward": env.task_completion_r,
                            # "Wipe/force_penalty": env.force_penalty,
                            # "Wipe/low_force_penalty":env.low_force_penalty,
                            "Wipe/wipe_contact_reward": env.wipe_contact_r,
                            # "Wipe/Return_unnormalized": Return_unnormalized,
                            # "Wipe/unnormalized_reward": env.un_normalized_reward,
                            'Wipe/vel_y': env.robots[0]._hand_vel[1],
                            "Wipe/unit_wiped_reward": env.unit_wipe,
                            # "Wipe/time_ratio":t_contact/t,

                            'Wipe/dist': dist,
                            'Wipe/ncontacts': env.sim.data.ncon,
                            }
            
            wandb.log({**metrics})

        
        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, force : {}".format(site, dist, Return, t, total_force))
            site = next(sites)

            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
            
        if done: #removed horizon limit
            print('done')
            break

    env.close()





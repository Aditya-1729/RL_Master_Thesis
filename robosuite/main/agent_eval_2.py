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
from robosuite.wrappers import RL_agent_2, GymWrapper, PolishingGymWrapper, ResidualWrapper
from stable_baselines3 import SAC


if __name__ == "__main__":
    
    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    if cfg.use_wandb:
        run = wandb.init(
        project='agent_eval',
        name=f"Curr",
        config=cfg) 
        config = wandb.config
    

    env = PolishingGymWrapper(suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller)))
                    
    
    #impedence mode and control delta

    # initialize the task

    dist_th = cfg.task_config.dist_th
    indent=cfg.task_config.indent
  
    env.reset()

    Return=0
    Return_unnormalized=0
    t=0
    t_contact=0

    model = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/21_9/robosuite/exp/Polishing/CRL_s_5_h_0.02_stp_10/2023.10.04/115331/best_model.zip")

    obs,_ = env.reset()

    Return=0
    t=0
    # do visualization

    while env.site!=env.objs[0].sites[-1]:
        action, _states = model.predict(observation=obs, deterministic=True)
        # print(f"action:{action}")
        total_force = np.linalg.norm(env.robots[0].ee_force - env.ee_force_bias)
        obs, reward, _, done, info = env.step(action)
        t+=1
        # env.render()

        # dist = np.linalg.norm(env.delta)
        Return +=reward
        if cfg.use_wandb:
            metrics = { 'Wipe/reward': reward,
                            # 'Wipe/kdy': action[1],
                            # 'Wipe/kpy': action[-2],
                            # 'Wipe/kpz': action[-3],
                            # 'Wipe/kpz': action[2],
                            'Wipe/Return': Return,
                            'Wipe/force': total_force,
                            'Wipe/vel_y': env.robots[0]._hand_vel[1],
                            # "Wipe/force_reward": env.force_in_window_reward,
                            "Wipe/task_complete_reward": env.task_completion_r,
                            "Wipe/wipe_contact_reward": env.wipe_contact_r,
                            # "Wipe/penalty": env.force_penalty,
                            "Wipe/unit_wiped_reward": env.unit_wipe,
                            'Wipe/dist': env.dist,}
            wandb.log({**metrics})

        # print(Return)
        if env.dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, total_force : {}".format(env.site, env.dist, Return, t, total_force))
        if t>2000 or env.task_completion_r or done:
            env.reset()
            break
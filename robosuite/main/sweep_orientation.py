import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.controllers import load_controller_config
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from itertools import cycle
import wandb
import numpy as np

if __name__ == "__main__":

    initialize_config_dir(version_base=None, config_dir="/hpcwork/ru745256/master_thesis/30_6/robosuite/robosuite/main/config")
    cfg = compose(config_name="main")
    
    env = GymWrapper(suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller)))

    default_config = OmegaConf.to_container(cfg)
    wandb.init(config=default_config,
               project="HPC_sweep")
    
    config = wandb.config
    dist_th = cfg.task_config.dist_th
    indent=cfg.task_config.indent

    model = SAC.load("/hpcwork/ru745256/master_thesis/logs/gpu_progress/best_model.zip")

    obs, _ = env.reset()

    Return=0
    t=0
    t_contact=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
 
    while site!=env.objs[0].sites[-1]:
        action, _states = model.predict(obs, deterministic=True)
        act = np.empty(15)
        a = site_pos
        t+=1

        act[:3] = action[:3]
        act[3:6] = config.Kd_or
        act[6:9]= action[:-3]
        act[9:12] = config.Kp_or
        act[-3:-1] = eef_pos[:2] + np.clip(a[:2]-eef_pos[:2], a_min=np.array([-0.01, -0.01]), a_max=np.array([0.01, 0.01]))
        act[-1] = eef_pos[-1] + np.clip(a[-1] - indent - eef_pos[-1], a_min = -0.01, a_max=0.01) 

        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        if total_force<5 and total_force>1:
            t_contact+=1
        total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))

        obs, reward, done, _, info = env.step(act)

        eef_pos = obs["robot0_eef_pos"]
        delta = eef_pos - site_pos

        dist = np.linalg.norm(delta)
        Return +=reward
        metrics = { 'Wipe/reward': reward,
                        'Wipe/Return': Return,
                        'Wipe/force': total_force,
                        "Wipe/task_complete_reward": env.task_completion_r,
                        "Wipe/wipe_contact_reward": env.wipe_contact_r,
                        "Wipe/penalty": env.force_penalty,
                        "Wipe/unit_wiped_reward": env.unit_wipe,
                        "Wipe/time_ratio":t_contact/t,
                        'Wipe/dist': dist,}
        
        wandb.log({**metrics})

        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, moment : {}".format(site, dist, Return, t, np.array(env.robots[0].recent_ee_forcetorques.current[:-3])))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if t>2000 or env.task_completion_r or done:
            env.reset()
            # env.close()
            break
    

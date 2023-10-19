# Manipulation environments
from robosuite.utils.input_utils import *
from itertools import cycle
import wandb
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import ResidualWrapper
from stable_baselines3 import SAC
from Residual_RL.src.policies import ResidualSAC
import hydra

@hydra.main(version_base=None, config_path="/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/DR/robosuite/robosuite/main/config", config_name="main")
def main(cfg: DictConfig):

    if cfg.use_wandb:
        run = wandb.init(
        # Set the project where this run will be logged
        project='Report',
        name=f"SAC_Residual_0.90",
        tags=['table'],
        config=cfg) 
  

    env=ResidualWrapper(suite.make(env_name=cfg.env.name,
                        **cfg.env.specs,
                        task_config=OmegaConf.to_container(
                            cfg.task_config),
                        controller_configs=OmegaConf.to_container(cfg.controller)))                  
    
    # initialize the task

    dist_th = cfg.task_config.dist_th
  
    env.reset()

#load the saved model SAC_residual    
    model = ResidualSAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/21_9/robosuite/Data_visualizer/exp/Residual_SAC/2023.10.07/s_5/204606/best_model.zip")
    
    obs,_ = env.reset()

    Return=0
    t=0
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    
    while site!=env.objs[0].sites[-2]:
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        action, _states = model.predict(observation=obs, deterministic=True)
        total_force = np.linalg.norm(env.robots[0].ee_force - env.ee_force_bias)
        obs, reward,done,_, info = env.step(action)
        t+=1
        # env.render()
        delta = eef_pos - site_pos
        dist = np.linalg.norm(delta)
        Return +=reward
        if cfg.use_wandb:
            metrics = { 'Wipe/reward': reward,
                            'Wipe/Return': Return,
                            'Wipe/force': total_force,
                            'Wipe/xdev': eef_pos[0]-site_pos[0],
                            "Wipe/task_complete_reward": env.task_completion_r,
                            "Wipe/wipe_contact_reward": env.wipe_contact_r,
                            'Wipe/vel_y': env.robots[0]._hand_vel[1],
                            "Wipe/unit_wiped_reward": env.unit_wipe,
                            'Wipe/dist': dist,
                            'Wipe/ncontacts': env.sim.data.ncon,
                            }
            
            wandb.log({**metrics})


        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, total_force : {}".format(site, dist, Return, t, total_force))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        
        if done:
            break

if __name__ == "__main__":
    main()
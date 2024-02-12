# Manipulation environments
from robosuite.utils.input_utils import *
from itertools import cycle
import json
import wandb
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import CurriculumWrapper, ResidualWrapper, GymWrapper, SplitWrapper, StandaloneWrapper
from stable_baselines3 import SAC
from Residual_RL.src.policies import ResidualSAC
import hydra
import os

@hydra.main(version_base=None, config_path="/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/DR/robosuite/robosuite/main/config", config_name="main")
def main(cfg: DictConfig):
    def find_zip_file(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".zip"):
                    return os.path.join(root, file)
    
    if cfg.use_wandb:
        run = wandb.init(
        project='ppt',
        name=f"Residual_0.9",
        tags=['baseline'],
        config=cfg) 
    
    base_env = suite.make(env_name=cfg.env.name,
                                **cfg.env.specs,
                                task_config=OmegaConf.to_container(
                                    cfg.task_config),
                                controller_configs=OmegaConf.to_container(cfg.controller))

    if cfg.algorithm=="sac_curriculum":
        env=CurriculumWrapper(base_env)

    if cfg.algorithm=="sac_residual":
        env=ResidualWrapper(base_env)
            
    if cfg.algorithm=="sac_split":
        env=SplitWrapper(base_env, cfg)
           
    else:
        env= StandaloneWrapper(base_env)

    # initialize the task

    dist_th = cfg.task_config.dist_th
    indent=cfg.task_config.indent
  
    env.reset()

    Return=0
    Return_unnormalized=0
    t=0
    t_contact=0
#SAC_curriculum
    # model = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/DR/exp/Curriculum_SAC/2023.10.07/s_2/200121/best_model.zip")
    # model = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/21_9/robosuite/Data_visualizer/exp/Curriculum_SAC/2023.10.18/s_9/014755/best_model.zip")
    #with lr
    # model = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/report/19_10/Curriculum_SAC/2023.10.19/s_9/174046/best_model.zip")
    #with DR:
    # model = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/report/saved_models/report/Curriculum_SAC/2023.10.23/s_5/205354/best_model.zip")
    # model = SAC.load(find_zip_file("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/report/saved_models/report/Curriculum_SAC/2023.10.23/s_54"))

#SAC_residual    
    model = ResidualSAC.load(find_zip_file("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/report/saved_models/report/Residual_SAC/2023.10.23/s_12"))
#SAC_only
    # model=SAC.load(find_zip_file("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/report/saved_models/report/SAC/2023.10.23/s_54"))    

#SAC_split
    # model=SAC.load(find_zip_file("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/report/saved_models/report/SAC_Split/2023.10.23/s_18"))
    
    obs,_ = env.reset()

    Return=0
    t=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    
    F_mean=[]
    energy=0
    
    while site!=env.objs[0].sites[-2]:
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        action, _states = model.predict(observation=obs, deterministic=True)
        # print(f"action:{action}")
        total_force = np.linalg.norm(env.robots[0].recent_ee_forcetorques.average[:3] - env.ee_force_bias)
        obs, reward,done,_, info = env.step(action)
        t+=1
        F_mean.append(total_force)
        energy += np.sum(env.robots[0].js_energy)
        # env.render()
        delta = eef_pos - site_pos
        dist = np.linalg.norm(delta)
        Return +=reward
        if cfg.use_wandb:
        # Return_unnormalized +=env.un_normalized_reward
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
                            'Wiped/markers': info["nwipedmarkers"],
                            'Wiped/energy': energy,
                            }
            
            wandb.log({**metrics})


        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, total_force : {}, f_mean: {}".format(site, dist, Return, t, total_force, np.mean(F_mean)))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        
        if done:
            break

if __name__ == "__main__":
    main()
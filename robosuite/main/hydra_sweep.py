# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
wandb.login()
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize


if __name__ == "__main__":

    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    # print(OmegaConf.to_yaml(cfg))

    default_config = OmegaConf.to_container(cfg)
    # wandb.init(config=default_config,
    #            project="test_sweep")
    # print(OmegaConf.to_yaml(cfg))
    config = wandb.config
    
    env = suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller))


    dist_th = cfg.task_config.dist_th
    indent=cfg.task_config.indent

    env.reset()

    Return=0
    t=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]

    action = np.array((3,3,3,3,3,3, 40,40,40,40, 40, 40, 0.15,-0.2,0.945))
    while site!=env.objs[0].sites[-1]:
        a = site_pos
        t+=1
        action[:2] = config.Kd_xy
        action[2] = config.Kd_z
        action[3:6] = config.Kd_xy
        action[6:8]= config.Kp_xy
        action[8] = config.Kp_z
        action[9:12] = config.Kp_xy
        action[12:14] = a[:2]
        action[-1] = a[-1] - indent

        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))

        obs, reward, done, info = env.step(action)

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
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, moment : {}".format(site, dist, Return, t, np.array(env.robots[0].recent_ee_forcetorques.current[:-3])))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if t>200 or env.task_completion_r:
            env.reset()
            # env.close()
            break

# Manipulation environments
import robosuite.utils.transform_utils as T
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
wandb.login()

if __name__ == "__main__":

    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")


    default_config = OmegaConf.to_container(cfg)
    # wandb.init(config=default_config,
    #            project="test_sweep")



    env = suite.make(env_name=cfg.env.name,
                     **cfg.env.specs,
                     task_config=OmegaConf.to_container(cfg.task_config),
                     controller_configs=OmegaConf.to_container(cfg.controller))

    dist_th = cfg.task_config.dist_th
    indent = cfg.task_config.indent

    env.reset()

    Return = 0
    t = 0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[8]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    # site_pos =array([ 0.15, -0.2 ,  0.91])
    # final_site_pos = array([0.15, 0.2, 0.91])
    or_mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    fixed_q = T.mat2quat(or_mat)  # (xyzw)

    while site != env.objs[0].sites[-1]:
        action = np.empty(19)
        a = site_pos
        eef_pos = env._eef_xpos
        a[:2] = eef_pos[:2] + np.clip(a[:2]-eef_pos[:2], a_min=np.array(
            [-0.01, -0.01]), a_max=np.array([0.01, 0.01]))
        a[-1] = eef_pos[-1] + \
            np.clip(a[-1] - indent - eef_pos[-1], a_min=-0.01, a_max=0.01)

        t += 1
        action[:2] = cfg.controller.damping_ratio
        action[2] = cfg.controller.damping_ratio
        action[3:6] = cfg.controller.damping_ratio
        action[6:8] = cfg.controller.kp
        action[8] = cfg.controller.kp
        action[9:12] = cfg.controller.kp
        action[12:14] = a[:2]
        action[14] = a[14] - indent
        action[-4:] = fixed_q
        total_force = np.linalg.norm(
            np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        total_moment = np.linalg.norm(
            np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))

        obs, reward, done, info = env.step(action)

        delta = eef_pos - site_pos

        dist = np.linalg.norm(delta)
        Return += reward
        metrics = {'Wipe/reward': reward,
                   'Wipe/Return': Return,
                   'Wipe/force': total_force,
                   }

        wandb.log({**metrics})

        # print(Return)
        # if dist < dist_th:
        #     print("wiped: {}, distance: {}, Return: {}, timesteps: {}, moment : {}".format(
        #         site, dist, Return, t, np.array(env.robots[0].recent_ee_forcetorques.current[:-3])))
        #     site = next(sites)
        #     site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if t > 2000 or env.task_completion_r:
            env.reset()
            # env.close()
            break

# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
import wandb
# wandb.login()
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from robosuite.wrappers import Via_points_full

if __name__ == "__main__":
    path = "/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/git_clean/robosuite/robosuite/controllers/config/position_polishing.json"
    # Notice how the environment is wrapped by the wrapper
    
    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    # run = wandb.init(
    # # Set the project where this run will be logged
    # project="test_nc_wrapper",
    # name='indent_subtraction',
    # config=cfg)
    
    # Track hyperparameters and run metadata
    # print(OmegaConf.to_dict(cfg))
    # initialize(version_base=None, config_path="config/")
    # cfg = compose(config_name="main")
    # default_config = OmegaConf.to_yaml(cfg)
    # config = wandb.config
    env = Via_points_full(
        suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller)),cfg
                    
    )

    env.reset(seed=0)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(2000):
            env.render()
            action = env.action_space.sample()
            # print(f"pre_controller:{action}")
            # print(f"eef_z:{env.sim.data.site_xpos[env.robots[0].eef_site_id][-1]}")
            observation, reward, terminated, truncated, info = env.step(action)
            # print(f"post_controller:{action}")
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                env.close()
                break

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
# wandb.login()
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base = None, config_path="config/", config_name="main")
def run(cfg: DictConfig):
    # wandb.config = OmegaConf.to_yaml(cfg)
    print(OmegaConf.to_yaml(cfg))
    env = suite.make(env_name=cfg.env.name,
                    **cfg.env.specs,
                    task_config = OmegaConf.to_container(cfg.task_config),
                    controller_configs=OmegaConf.to_container(cfg.controller))
                    
    print(env.arm_limit_collision_penalty)
    low, high = env.action_spec
    for i in range(1000):
    # action = np.random.uniform(low, high)
        # action=np.array([0, -0.05, -0.1])
        action = np.random.uniform(low, high)
        
        # action[:3] = [0,0,-0.001]
        # action[:-4]: np.array([0.3,0, 0, -0.9])
        obs, reward, done, _ = env.step(action)
        # print(obs["robot0_eef_pos"][2])
        env.render()
        # if i%500 ==0:
        #     env.reset()
    env.reset()
    env.close()
if __name__=="__main__":
    run()

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
# from time_curriculum.src.tbc.tbc import get_tbc_algorithm
import robosuite as suite
from robosuite.wrappers import Via_points_full
from residual_rl.src.policies import ResidualSAC, SACResidualPolicy
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
import os


# scriptDir = os.path.dirname(os.path.realpath(__file__))
# @hydra.main(version_base=None, \
#             config_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'robosuite/main/config')),\
#             config_name="main")
@hydra.main(version_base=None, \
            config_path='/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/1_8/robosuite/robosuite/main/config',\
            config_name="main")
def main(cfg: DictConfig):
    
    env=Via_points_full(suite.make(env_name=cfg.env.name,
                            **cfg.env.specs,
                            task_config=OmegaConf.to_container(
                                cfg.task_config),
                            controller_configs=OmegaConf.to_container(cfg.controller)),cfg)    
    # guide_policy = ResidualSAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/saved_models/4N.zip").policy
    n = 5
    max_horizon = env.env.horizon
    MlpPolicy = SACResidualPolicy
    #percentage of training used up for handover
    p=0.1

    model = ResidualSAC(
        env=env,
        **cfg.algorithm.model,
    )

    
    callbacks = [EvalCallback(eval_env=env, **cfg.algorithm.eval)]
    
    model.learn(**cfg.algorithm.learn, callback=callbacks)
    

if __name__ == "__main__":
    main()

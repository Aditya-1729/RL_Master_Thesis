# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
# wandb.login()
from omegaconf import DictConfig, OmegaConf
from robosuite.wrappers import Nominal_controller_gym
# from robosuite.controllers import load_controller_config
from hydra import compose, initialize

if __name__ == "__main__":
    path = "/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/git_clean/robosuite/robosuite/controllers/config/position_polishing_var.json"

    # initialize(version_base=None, config_path="config/")
    # cfg = compose(config_name="main")
# def objective(config):
#     path = "/home/aditya/robosuite/robosuite/controllers/config/position_polishing_var.json"
#     with open(path) as f:
#         controller_config = json.load(f)

    # initialize the task
    env = suite.make(
        env_name= "Polishing",
        robots= "Panda",
        controller_configs= load_controller_config(custom_fpath=path),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20
    )
    # env = Nominal_controller_gym(
    #     suite.make(env_name=cfg.env.name,
    #                 **cfg.env.specs,
    #                 task_config = OmegaConf.to_container(cfg.task_config),
    #                 controller_configs=OmegaConf.to_container(cfg.controller))
    #                 ,cfg
    # )


# writer = SummaryWriter('/home/adityapradhan/robosuite/robosuite/Runs/force_kp_{kp}_kd_{kd}_th_{th}_in_{ind}'.format(kp=controller_config['kp'], \
#                         kd=controller_config['damping_ratio'], th=dist_th, ind=indent))




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
    action[:2] = 3
    action[2] = 3
    action[2:4] = 40
    action[5] = 40
    action[6:8] = a[:2]
    action[-1] = a[-1] - 0.003




    # total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
    # total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))

    obs, reward, done, info = env.step(action)

#     eef_pos = obs["robot0_eef_pos"]
#     delta = eef_pos - site_pos

#     dist = np.linalg.norm(delta)
#     Return +=reward
#     metrics = { 'Wipe/reward': reward,
#                     'Wipe/Return': Return,
#                     'Wipe/force': total_force,
#                     "Wipe/force_reward": env.force_in_window_reward,
#                     "Wipe/task_complete_reward": env.task_completion_r,
#                     "Wipe/wipe_contact_reward": env.wipe_contact_r,
#                     "Wipe/penalty": env.force_penalty,
#                     "Wipe/unit_wiped_reward": env.unit_wipe,
#                     "Wipe/moment":total_moment,
#                     'Wipe/dist': dist,}
    
#     # wandb.log({**metrics})

    
#     # print(Return)
#     if dist < dist_th:
#         print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, moment : {}".format(site, dist, Return, t, np.array(env.robots[0].recent_ee_forcetorques.current[:-3])))
#         site = next(sites)
#         site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
#     if t>2000 or env.task_completion_r:
#         env.reset()
#         break


# def main():
#     wandb.init(project='my-first-sweep')
#     Return, reward, total_force = objective(wandb.config)
#     wandb.log({'Return': Return,
#                'reward': reward,
#                'contact_force': total_force,
#                })

# sweep_configuration = {
#     'method': 'random',
#     'metric': {'goal': 'maximize', 'name': 'Return'},
#     'parameters': 
#     {
#         'Kp_xy': {'distribution':'int_uniform', 'max': 300, 'min': 0},
#         'Kp_z': {'distribution':'int_uniform', 'max': 300, 'min': 0},
#         'Kd_xy': {'distribution':'int_uniform', 'max': 10, 'min': 0},
#         'Kd_z': {'distribution':'int_uniform', 'max': 10, 'min': 0}},
#     }



# sweep_id = wandb.sweep(
# sweep=sweep_configuration, 
# project='my-first-sweep'
# )


# # env.close()

# wandb.agent(sweep_id,function=main, count=10)
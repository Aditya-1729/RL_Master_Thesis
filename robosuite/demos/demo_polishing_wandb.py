# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json
import wandb
# wandb.login()


if __name__ == "__main__":
    
    run = wandb.init(
    # Set the project where this run will be logged
    project="nominal_control_log",
    notes="force offset added")
    # Track hyperparameters and run metadata

    #impedence mode and control delta
    path = "/home/aditya/robosuite/robosuite/controllers/config/position_polishing.json"
    with open(path) as f:
        controller_config = json.load(f)

    # initialize the task
    env = suite.make(
        env_name= "Polishing",
        robots= "Panda",
        controller_configs= load_controller_config(custom_fpath=path),
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20
    )
    dist_th = 0.01
    indent=0.003
    # writer = SummaryWriter('/home/adityapradhan/robosuite/robosuite/Runs/force_kp_{kp}_kd_{kd}_th_{th}_in_{ind}'.format(kp=controller_config['kp'], \
    #                         kd=controller_config['damping_ratio'], th=dist_th, ind=indent))
    env.reset()

    Return=0
    t=0
    t_contact=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    action = np.array((0.15,-0.2,0.945))
    while site!=env.objs[0].sites[-1]:
        a = site_pos
        t+=1
        # action[:2] = a[:2]
        # action[-1] = a[-1] - indent
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        # action[:2] = a[:2] 
        action[:2] = eef_pos[:2] + np.clip(a[:2]-eef_pos[:2], a_min=np.array([-0.01, -0.01]), a_max=np.array([0.01, 0.01]))
        action[-1] = eef_pos[-1] + np.clip(a[-1] - indent - eef_pos[-1], a_min = -0.01, a_max=0.01) 
        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        # total_moment = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:-3]))
        if total_force<5 and total_force>1:
            t_contact+=1
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
                        "Wipe/force_penalty": env.force_penalty,
                        "Wipe/wipe_contact_reward": env.wipe_contact_r,
                        # "Wipe/total_dist_reward": env.total_dist_reward,
                        # "Wipe/unnormalized_reward": env.un_normalized_reward,
                        "Wipe/unit_wiped_reward": env.unit_wipe,
                        "Wipe/time_ratio":t_contact/t,
                        'Wipe/dist': dist,
                        'Wipe/ncontacts': env.sim.data.ncon,
                        }
        
        wandb.log({**metrics})

        
        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, moment : {}".format(site, dist, Return, t, np.array(env.robots[0].recent_ee_forcetorques.current[:-3])))
            site = next(sites)
            # print(f"robot_joint_pos: {obs['robot0_joint_pos']}")
            print(f"robot_joint_pos: {env.robots[0]._joint_positions}")
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
        if env.task_completion_r: #removed horizon limit
            env.reset()
            break
    wandb.finish()





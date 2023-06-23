# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import json

if __name__ == "__main__":
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
    indent=0.001
    writer = SummaryWriter('/home/aditya/robosuite/robosuite/runs/kp_{kp}_kd_{kd}_th_{th}_in_{ind}'.format(kp=controller_config['kp'], \
                            kd=controller_config['damping_ratio'], th=dist_th, ind=indent))
    # env.reset()
    # env.viewer.set_camera(camera_id=0)

    # # Get action limits
    # low, high = env.action_spec

    # # do visualization
    # for i in range(1000):
    #     action = np.random.uniform(low, high)
    #     obs, reward, done, _ = env.step(action)
    #     env.render()
    
    env.reset()
    # env.viewer.set_camera(camera_id=0)
    # Get action limits
    # low, high = env.action_spec
    # contact=False
    # print(env.sim.data.site_xpos[env.robots[0].eef_site_id])
    # print(env.sim.data.body_xpos)
    # for marker in env.model.mujoco_arena.markers:
        # print(markers)
    # print(env.mujoco_objects)
        # site_name = env.model.mujoco_arena.marker.sites)
    # print(env.robots[0].robot_joints)
    # site_id = env.sim.model.site_name2id(env.model.mujoco_arena.marker.sites[0])
    # print(site_id, env.model.mujoco_arena.marker.sites[0], env.sim.data.site_xpos[site_id])
    # env.sim.data.site_xpos[env.sim.model.site_name2id(env.model.mujoco_arena.marker.bottom_site)]
    # print(np.array(env.sim.data.site_xpos[env.sim.model.site_name2id(CurvedSurfaceObject.root_body)]))
    # print(low,high)    
    Return=0
    t=0
    # do visualization
    sites = cycle(env.objs[0].sites)
    site = env.objs[0].sites[0]
    site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
    action = np.array((0.15,-0.2,0.945))
    while site!=env.objs[0].sites[-1]:
        a = site_pos
        t+=1
        action[:2] = a[:2]
        action[-1] = a[-1] - indent

        for i in range(env.sim.data.ncon):
    # Note that the contact array has more than `ncon` entries,
    # so be careful to only read the valid entries.
            contact = env.sim.data.contact[i]
            print('contact', i)
            print('dist', contact.dist)
            print('geom1', contact.geom1, env.sim.model.geom_id2name(contact.geom1))
            print('geom2', contact.geom2, env.sim.model.geom_id2name(contact.geom2))
            # geom2_body = env.sim.model.geom_bodyid[env.sim.data.contact[i].geom2]
            # print(' Contact force on geom2 body', env.sim.data.cfrc_ext[geom2_body])
            # print('norm', np.sqrt(np.sum(np.square(env.sim.data.cfrc_ext[geom2_body]))))
            # writer.add_scalar('wipe/contact_force{g}'.format(g=geom2_body), np.sqrt(np.sum(np.square(env.sim.data.cfrc_ext[geom2_body]))), t)
        # action = site_pos
        # print(action, site_pos)
        # print(site_pos)
        # action = site_pos
        # current_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        # action[:2] = site_pos[:2]
        # action[-1] = site_pos[-1]-0.01
        # for site in env.objs[0].sites:

        # print(action)


        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        # if total_force<1 and contact==False: 
        #     action = np.array([0.08,-0.25,-0.15,0,0,0])
        # else:
        #     contact=True    
        #     action = np.array([0,0.3,-0.1,0,0,0])
        #     if total_force>18:
        #         action = np.array([0,0.1,0.1,0,0,0])
        # print(action.dtype())
        obs, reward, done, info = env.step(action)

        # print(site_pos)
        eef_pos = obs["robot0_eef_pos"]
        delta = eef_pos - site_pos
        # print(obs["robot0_eef_pos"][2], delta[2])
        dist = np.linalg.norm(delta)
        Return +=reward
        env.render()
        # writer.add_scalar('Wipe/reward', reward, t)
        # writer.add_scalar('Wipe/Return', Return, t)
        # writer.add_scalar('Wipe/force', total_force, t)
        # writer.add_scalar('Wipe/dist', dist, t)
        # writer.add_scalar('Wipe/x', obs["robot0_eef_pos"][0], t)
        # writer.add_scalar('Wipe/y', obs["robot0_eef_pos"][1], t)
        # writer.add_scalar('Wipe/z', obs["robot0_eef_pos"][2], t)
        
        # print(Return)
        if dist < dist_th:
            print ("wiped: {}, distance: {}, Return: {}, timesteps: {}, force : {}".format(site, dist, Return, t, np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))))
            site = next(sites)
            site_pos = env.sim.data.site_xpos[env.sim.model.site_name2id(site)]
            # action[:2] = site_pos[:2]
            # action[-1] = site_pos[-1] - 0.01
            # print("successfuly wiped site {}".format(site))
        # print(obs)
        # print(env.sim.data.site_xpos[env.sim.model.site_name2id(env.objs[0].sites[0])])
        # print("force:",env.robots[0].ee_force)
        # env.render()
        # if i%10==0:
            # env.reset()
            # contact=False
        # print("obs: {}, total_force {}, site_pos {}".format (obs["robot0_eef_pos"], total_force, site_pos))
        # print("obs: {}, total_force {}".format (obs["robot0_eef_pos"], total_force))
    env.reset()
    env.close()
    
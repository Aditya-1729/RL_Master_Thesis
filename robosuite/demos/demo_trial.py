# Manipulation environments
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

if __name__ == "__main__":


    # initialize the task
    env = suite.make(
        env_name= "CustomWipe",
        robots= "Panda",
        controller_configs= load_controller_config(custom_fpath='/home/aditya/robosuite/robosuite/controllers/config/osc_position.json'),
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20
    )


    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(1000):
        # action = np.random.uniform(low, high)
        action=np.array([0, -0.05, -0.1])
        obs, reward, done, _ = env.step(action)
        print(obs["robot0_eef_pos"][2])
        env.render()
''' 
    env.reset()
    # env.viewer.set_camera(camera_id=0)
    # Get action limits
    low, high = env.action_spec
    contact=False
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

    # do visualization
    for i in range(10000):
        total_force = np.linalg.norm(np.array(env.robots[0].recent_ee_forcetorques.current[:3]))
        if total_force<1 and contact==False: 
            action = np.array([0.08,-0.25,-0.15,0,0,0])
        else:
            contact=True    
            action = np.array([0,0.3,-0.1,0,0,0])
            if total_force>18:
                action = np.array([0,0.1,0.1,0,0,0])
        # print(action.dtype())
        obs, reward, done, _ = env.step(action)
        print(obs)
        
        # print("force:",env.robots[0].ee_force)
        env.render()
        # if i%10==0:
            # env.reset()
            # contact=False
        print("obs: {}, total_force {}".format (obs["robot0_eef_pos"], total_force))
    '''

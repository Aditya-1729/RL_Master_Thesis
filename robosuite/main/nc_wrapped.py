"""
This script shows how to adapt an environment to be compatible
with the Gymnasium API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Basic Usage" section of the Gymnasium documentation

The following snippet was used to demo basic functionality.

    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            env.close()

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

import robosuite as suite
from robosuite.wrappers import Nominal_controller_gym
from robosuite.controllers import load_controller_config
from hydra import compose, initialize


if __name__ == "__main__":
    path = "/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/git_clean/robosuite/robosuite/controllers/config/position_polishing.json"
    # Notice how the environment is wrapped by the wrapper
    
    initialize(version_base=None, config_path="config/")
    cfg = compose(config_name="main")
    
    env = Nominal_controller_gym(
        suite.make(
            "Polishing",
            robots="Panda",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,
            controller_configs= load_controller_config(custom_fpath=path),  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        ),cfg
    )

    env.reset(seed=0)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            print(f"pre_controller:{action}")
            print(f"eef_z:{env.sim.data.site_xpos[env.robots[0].eef_site_id][-1]}")
            observation, reward, terminated, truncated, info = env.step(action)
            # print(f"post_controller:{action}")
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                env.close()
                break

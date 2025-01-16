import gym
import isaacgymenvs
import torch

num_envs = 1

envs = isaacgymenvs.make(
    seed=0,
    task="Ant",
    num_envs=num_envs,
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
    # params for recording video
    headless=False,
    virtual_screen_capture=True,
    force_render=False,
)
envs.is_vector_env = True

# wrapper for recording video
# envs = gym.wrappers.RecordVideo(
#     envs,
#     './videos',
#     step_trigger=lambda x: x % 10000 == 0,  # record every 10000 steps
#     video_length=100,  # record 100 steps per video
# )

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
print("Initial observation is", obs.keys())

for _ in range(20000):
    random_actions = (
        2.0 * torch.rand((num_envs,) + envs.action_space.shape, device="cuda:0") - 1.0
    )
    envs.step(random_actions)

"""
for libx264 error:
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
"""

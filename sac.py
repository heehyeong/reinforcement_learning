import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from environment import env  # Assuming this imports your custom environment

# Define the SAC model
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4, 
    buffer_size=1_000_000, 
    batch_size=256, 
    tau=0.005, 
    gamma=0.99, 
    train_freq=1, 
    gradient_steps=1,
    ent_coef="auto"  # Automatically tune the entropy coefficient
)

# Train the model
model.learn(total_timesteps=1_000_000)  # Train for 1 million steps

# Save the trained model
model.save("sac_ant")

# Load the trained model
model = SAC.load("sac_ant")

# Wrap the environment for video recording
video_folder = "./videos/"
env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)  # Record every episode

# Reset the environment
obs = env.reset()

# Run the evaluation episode
for step in range(1000):  # Evaluation for 1000 steps
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

# Close the environment and ensure video is properly finalized
env.close()
print(f"Video recorded and saved in {video_folder}")

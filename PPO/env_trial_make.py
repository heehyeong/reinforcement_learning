import gymnasium
import numpy as np

class CustomRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Customize the reward
        x_position = info.get("x_position", 0)  # Position in the X-axis
        y_position = info.get("y_position", 0)  # Position in the Y-axis

        forward_reward = x_position  # Reward for moving forward
        lateral_penalty = -abs(y_position)  # Penalty for lateral deviations
        stability_penalty = -np.sum(np.abs(obs))  # Penalty for oscillations

        # New reward function
        new_reward = 2 * forward_reward + 0.5 * lateral_penalty + 0.1 * stability_penalty

        return obs, new_reward, done, truncated, info

def create_env():
    # Create and return the configured environment
    env = gymnasium.make(
        'Ant-v5',
        xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
        forward_reward_weight=1.5,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        main_body=1,
        healthy_z_range=(0.195, 0.75),
        reset_noise_scale=0.1,  # reset_noise_scale=0.0 for a deterministic environment
        frame_skip=25,
        max_episode_steps=1000,
        render_mode='human',
        width=1280,
        height=720,
    )
    return CustomRewardWrapper(env)


env_trial = create_env()
obs, info = env_trial.reset()
done = False
while not done:
    action = env_trial.action_space.sample()
    obs, reward, done, truncated, info = env_trial.step(action)
    print(f"Modified reward: {reward}")
    env_trial.render()
    if done or truncated:
        print("End of episode.")
        break
env_trial.close()
print("Observation space:", env_trial.observation_space)
print("Action space:", env_trial.action_space)

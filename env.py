import gymnasium 

def create_env():
    # Create and return the default environment
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
        render_mode='rgb_array',
    )
    return env

# Test code
if __name__ == "__main__":
    env = create_env()

    # Reset the environment
    obs, info = env.reset()
    done = False

    # Run a test episode
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        # Render the environment (if a viewer is available)
        try:
            env.render()
        except Exception as e:
            print(f"Render error: {e}")
        if done or truncated:
            print("End of episode.")
            break

    env.close()

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from env import create_env  # Import the environment from env.py

# Curriculum Learning Callback
class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 100_000 == 0:
            env = self.training_env.envs[0]
            if hasattr(env, "healthy_z_range"):
                new_range = max(env.healthy_z_range[0] - 0.01, 0.195)
                env.healthy_z_range = (new_range, 0.75)
                print(f"Updated healthy_z_range to {env.healthy_z_range}")
            else:
                print("Warning: 'healthy_z_range' attribute not found in the environment.")
        return True

# Directory for saving checkpoints
checkpoint_dir = './checkpoints_sac/'

# Create the environment (without RecordVideo)
env = create_env()

# Initialize the SAC model with custom hyperparameters
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    batch_size=512,
    gamma=0.98,
    verbose=1
)

# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="sac_checkpoint"
)

# Curriculum Learning Callback
curriculum_callback = CurriculumCallback()

# Combine Callbacks
callback = CallbackList([checkpoint_callback, curriculum_callback])

# Train the SAC agent with Curriculum Learning
model.learn(total_timesteps=1_000_000, callback=callback)

# Save the final trained model
model.save("sac_ant_curriculum")

# Record a video of the trained agent
video_dir = "./videos/"
eval_env = create_env()  # Create a new environment for evaluation
eval_env = RecordVideo(eval_env, video_folder=video_dir, episode_trigger=lambda episode_id: True)  # Record all episodes

# Reset the environment
obs, info = eval_env.reset()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    # Take a step in the environment
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

eval_env.close()

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from environment import env

# PPO 모델 정의
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.000974386667832731, 
    n_steps=1024, 
    batch_size=160, 
    n_epochs=10,
    gamma=0.9338134593986138, 
    gae_lambda=0.95, 
    clip_range=0.2, 
    ent_coef=1.966049945383333e-05,
    tensorboard_log="./ppo_tensorboard/"
)

# 학습
model.learn(total_timesteps=1_000_000)  # 100만 스텝 학습

# 학습된 모델 저장
model.save("ppo_ant")

# 모델 로드 후 평가
model = PPO.load("ppo_ant")
obs = env.reset()

for step in range(1000):  # 평가 에피소드
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
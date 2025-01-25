import os
import json
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import optuna
from optuna.samplers import TPESampler

# Optuna 최적화 목표 함수
def objective(trial):
    # 하이퍼파라미터를 Optuna에서 샘플링
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    n_steps = trial.suggest_int("n_steps", 1024, 4096, step=512)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)

    # 환경 초기화
    env = make_vec_env("Ant-v5", n_envs=1)

    # PPO 모델 초기화
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
    )

    # 모델 학습
    model.learn(total_timesteps=50000)

    # 평가 점수 계산
    mean_reward = evaluate_model(model, env)
    env.close()

    return mean_reward

# 평가 함수
def evaluate_model(model, env, n_eval_episodes=5):
    mean_reward = 0.0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        mean_reward += episode_reward
    return mean_reward / n_eval_episodes

os.system("rm -rf ./logs/")
os.system("mkdir -p ./optuna/")

# Optuna 설정 및 학습
seed = 42
storage_file = "sqlite:///optuna/ppo_ant.db"  # SQLite로 저장
study_name = "ppo_ant"
study_dir = "optuna_results"
full_study_dir_path = f"optuna/{study_name}"

# Optuna Study 생성
sampler = TPESampler(seed=seed)
study = optuna.create_study(
    sampler=sampler,
    direction="maximize",
    study_name=study_name,
    storage=storage_file,
    load_if_exists=True
)

# 최적화 실행
n_trials = 20
print(f"Running {n_trials} trials for PPO optimization...")
study.optimize(objective, n_trials=n_trials)

# 최적의 결과 저장
best_trial = study.best_trial
best_params = best_trial.params

# JSON 파일로 저장
output_file = os.path.join(study_dir, "best_hyperparameters.json")
with open(output_file, "w") as f:
    json.dump(best_params, f, indent=4)

print(f"Best hyperparameters saved to {output_file}")

# Generate the improtant figures of the results
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(f"{study_dir}/optimization_history.html")
fig = optuna.visualization.plot_contour(study)
fig.write_html(f"{study_dir}/contour.html")
fig = optuna.visualization.plot_slice(study)
fig.write_html(f"{study_dir}/slice.html")
fig = optuna.visualization.plot_param_importances(study)
fig.write_html(f"{study_dir}/param_importances.html")
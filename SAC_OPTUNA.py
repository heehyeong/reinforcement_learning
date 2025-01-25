import os
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import create_env  # Import the environment from env.py

# Create the environment
def create_wrapped_env():
    env = Monitor(create_env())  # Monitor for tracking rewards
    env = DummyVecEnv([lambda: env])  # DummyVecEnv for compatibility
    return env

# Objective function for Optuna
def objective(trial):
    env = create_wrapped_env()

    # Suggest hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    tau = trial.suggest_float('tau', 0.005, 0.02)
    ent_coef = trial.suggest_float('ent_coef', 1e-3, 1e-1, log=True)


    # Initialize the model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        verbose=0,
    )

    # Evaluation callback to compute reward during training
    eval_env = create_wrapped_env()
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    # Train the model
    model.learn(total_timesteps=100_000, callback=eval_callback)

    # Evaluate the model
    mean_reward = eval_callback.best_mean_reward
    env.close()
    eval_env.close()
    return mean_reward  # Optuna will maximize this value

# Run Optuna optimization
if __name__ == "__main__":
    # Configure Optuna with storage and pruner
    study = optuna.create_study(
        direction="maximize",
        study_name="SAC_Optimization",
        storage="sqlite:///sac_optuna.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10_000),
    )

    # Run optimization
    study.optimize(objective, n_trials=20)  # Run 50 trials

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)

    # Train a new model with the best hyperparameters
    best_params = study.best_params
    env = create_wrapped_env()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        gamma=best_params['gamma'],
        tau=best_params['tau'],
        ent_coef=best_params['ent_coef'],
        verbose=1,
    )

    # Train the final model
    model.learn(total_timesteps=1_000_000)

    # Save the final trained model
    model.save("sac_ant_optuna")

    # Visualize results
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()

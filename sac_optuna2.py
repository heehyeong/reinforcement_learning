import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import optuna.visualization as vis

def objective(trial):
    # Define the environment
    env_name = "Ant-v5"  # environment name
    env = make_vec_env(env_name, n_envs=1)

    # Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int("buffer_size", 10000, 200000)
    batch_size = trial.suggest_int("batch_size", 64, 256)
    tau = trial.suggest_float("tau", 0.005, 0.02)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    train_freq = trial.suggest_int("train_freq", 8, 64)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 64)

    # Create the SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=0,
    )

    # Train the model
    model.learn(total_timesteps=10000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    # Close the environment
    env.close()

    return mean_reward

if __name__ == "__main__":
    # Create the Optuna study
    study = optuna.create_study(direction="maximize")

    # Run optimization
    study.optimize(objective, n_trials=50)

    # Print best trial details
    print("Best trial:")
    best_trial = study.best_trial
    print(f"Value: {best_trial.value}")
    print("Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save plots as HTML files
    vis.plot_optimization_history(study).write_html("optimization_history.html")
    vis.plot_param_importances(study).write_html("param_importances.html")
    vis.plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
    vis.plot_slice(study).write_html("slice_plot.html")
    vis.plot_contour(study).write_html("contour_plot.html")

    print("Plots have been saved as HTML files.")

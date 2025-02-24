import numpy as np
import matplotlib.pyplot as plt

# Define the environments
environments = ["HalfCheetah-v4", "Pendulum-v1", "Hockey_Weak_Bot", "Hockey_Strong_Bot"]
window_size = 10  # For smoothing training rewards

training_files_dict = {
    "HalfCheetah-v4": ["./results/TD3__HalfCheetah-v4__100_training_rewards.npy",
                       "./results/TD3__HalfCheetah-v4__104_training_rewards.npy"],
    "Pendulum-v1": ["./results/TD3__Pendulum-v1__101_training_rewards.npy",
                    "./results/TD3__Pendulum-v1__105_training_rewards.npy"],
    "Hockey_Weak_Bot": ["./results/TD3__HockeyEnv__25_training_rewards.npy",
                        "./results/TD3__HockeyEnv__23_training_rewards.npy"],
    "Hockey_Strong_Bot": ["./results/TD3__HockeyEnv__26_training_rewards.npy",
                          "./results/TD3__HockeyEnv__24_training_rewards.npy"]
}

evaluation_files_dict = {
    "HalfCheetah-v4": ["./results/TD3__HalfCheetah-v4__100.npy",
                       "./results/TD3__HalfCheetah-v4__104.npy"],
    "Pendulum-v1": ["./results/TD3__Pendulum-v1__101.npy",
                    "./results/TD3__Pendulum-v1__105.npy"],
    "Hockey_Weak_Bot": ["./results/TD3__HockeyEnv__25.npy",
                        "./results/TD3__HockeyEnv__23.npy"],
    "Hockey_Strong_Bot": ["./results/TD3__HockeyEnv__26.npy",
                          "./results/TD3__HockeyEnv__24.npy"]
}

training_labels_dict = {
    "./results/TD3__HalfCheetah-v4__100_training_rewards.npy": "Basic TD3",
    "./results/TD3__HalfCheetah-v4__104_training_rewards.npy": "Modified TD3",
    "./results/TD3__Pendulum-v1__101_training_rewards.npy": "Basic TD3",
    "./results/TD3__Pendulum-v1__105_training_rewards.npy": "Modified TD3",
    "./results/TD3__HockeyEnv__23_training_rewards.npy": "Modified TD3",
    "./results/TD3__HockeyEnv__25_training_rewards.npy": "Basic TD3",
    "./results/TD3__HockeyEnv__24_training_rewards.npy": "Modified TD3",
    "./results/TD3__HockeyEnv__26_training_rewards.npy": "Basic TD3",
}

evaluation_labels_dict = {
    "./results/TD3__HalfCheetah-v4__100.npy": "Basic TD3",
    "./results/TD3__HalfCheetah-v4__104.npy": "Modified TD3",
    "./results/TD3__Pendulum-v1__101.npy": "Basic TD3",
    "./results/TD3__Pendulum-v1__105.npy": "Modified TD3",
    "./results/TD3__HockeyEnv__23.npy": "Modified TD3",
    "./results/TD3__HockeyEnv__25.npy": "Basic TD3",
    "./results/TD3__HockeyEnv__24.npy": "Modified TD3",
    "./results/TD3__HockeyEnv__26.npy": "Basic TD3",
}


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


for env in environments:
    training_files = training_files_dict.get(env, [])
    evaluation_files = evaluation_files_dict.get(env, [])

    plt.figure(figsize=(6, 4))
    for file in training_files:
        try:
            training_rewards = np.load(file)
            smoothed_rewards = moving_average(training_rewards, window_size)
            label = training_labels_dict.get(file, file)
            plt.plot(smoothed_rewards, label=label)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Reward")
    plt.title(f"Training Reward - {env}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    for file in evaluation_files:
        try:
            evaluations = np.load(file)
            label = evaluation_labels_dict.get(file, file)
            plt.plot(evaluations, label=label)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    plt.xlabel("Evaluation Number")
    plt.ylabel("Average Reward")
    plt.title(f"Evaluation Reward - {env}")
    plt.legend()
    plt.grid(True)

    # Save the evaluation plots
    eval_plot_filename = f"./results/evaluation_{env}.png"
    plt.savefig(eval_plot_filename)
    print(f"Saved evaluation plot: {eval_plot_filename}")

    plt.show()

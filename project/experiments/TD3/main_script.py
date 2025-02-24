import numpy as np
import torch
from hockey.hockey_env import HockeyEnv, BasicOpponent
import gymnasium as gym
import os
from TD3_script import TD3, PinkNoise


def evaluate_policy(policy, env_seed, env_name, episodes=10):
    if env_name == "Hockey":
        evaluation_env = HockeyEnv()
        opponent = BasicOpponent(weak=False)
    else:
        evaluation_env = gym.make(env_name)

    avg_reward = 0.
    for _ in range(episodes):
        state, _ = evaluation_env.reset(seed=env_seed)
        done = False
        while not done:
            action_p1 = policy.select_action(np.array(state))  # TD3 agent
            if env_name == "Hockey":
                action_p2 = opponent.act(evaluation_env.obs_agent_two())  # Opponent
                state, reward, done, truncated, info = evaluation_env.step(np.hstack([action_p1, action_p2]))
            else:
                state, reward, done, truncated, info = evaluation_env.step(action_p1)

            done = done or truncated
            avg_reward += reward

    avg_reward /= episodes
    print("---------------------------------------")
    print(f"Evaluation over {episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == '__main__':
    # Set the seeds
    seed = 109
    env_seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize the environment
    env_name = "Hockey"
    if env_name == "Hockey":
        env = HockeyEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = 4
        max_action = float(env.action_space.high[0])
        opponent = BasicOpponent(weak=False)
    elif env_name == "HalfCheetah-v4":
        env = gym.make("HalfCheetah-v4")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
    elif env_name == "Pendulum-v1":
        env = gym.make("Pendulum-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    # Create the necessary folders
    filename = f"TD3__{env_name}__{seed}"
    save = True
    load = False
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if save and not os.path.exists("./models"):
        os.makedirs("./models")

    # Define the policy
    learning_rate = 0.001
    policy = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, learning_rate=learning_rate)
    if load:
        policy_file = filename
        policy.load(f"./models/{policy_file}")

    # Initialize the optimizers for actor and critic
    actor_optimizer = policy.actor_optimizer
    critic_optimizer = policy.critic_optimizer

    # Initialize the learning rate scheduler for both optimizers
    actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.99)
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.99)

    print("---------------------------------------")
    print(f"TD3 {env_name} Environment Seed {seed}")
    print("---------------------------------------")

    # Reset the parameters before running the TD3 algorithm
    state, _ = env.reset(seed=env_seed)
    done = False
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0
    training_rewards = []

    # Define the needed parameters
    max_time_steps = 5e5  # Number of time steps that the script is executed for
    untrained_time_steps = 25e3  # Number of time steps where the initial random policy is used
    exploration_noise_std = 1.0  # Define the Gaussian distribution for the exploration noise
    eval_freq = 5e3  # Evaluate the policy at these intervals
    pink_noise = PinkNoise(action_dim)  # Define pink noise

    # Evaluate the untrained policy
    evaluations = [evaluate_policy(policy, env_seed, env_name)]

    for t in range(int(max_time_steps)):
        episode_time_steps += 1

        # Select an action randomly or according to policy
        if t < untrained_time_steps:
            if env_name == "Hockey":
                action_p1 = np.random.uniform(-max_action, max_action, action_dim).tolist()
            else:
                action_p1 = env.action_space.sample().tolist()
        else:
            action_p1 = (policy.select_action(np.array(state))
                         + pink_noise.sample() * exploration_noise_std
                         ).clip(-max_action, max_action)

        # Perform the action
        if env_name == "Hockey":
            action_p2 = opponent.act(env.obs_agent_two())
            next_state, reward, done, truncated, info = env.step(np.hstack([action_p1, action_p2]))
        else:
            next_state, reward, done, truncated, info = env.step(action_p1)

        done = done or truncated

        # Store the data in the replay buffer
        policy.replay_buffer.add(state, action_p1, next_state, reward, done)

        state = next_state
        episode_reward += reward

        # Train the agent after collecting enough data
        if t >= untrained_time_steps:
            policy.train()

        if done:
            training_rewards.append(episode_reward)
            print(
                f"Time step: {t + 1} Episode: {episode_num + 1} Episode time steps: {episode_time_steps} Reward: {episode_reward:.3f}")

            # Update the learning rate using the schedulers
            if episode_num % 20 == 0 and episode_num != 0 and t >= untrained_time_steps:
                actor_scheduler.step()
                critic_scheduler.step()
                policy.update_optimizer_lr(actor_optimizer.param_groups[0]['lr'],
                                           critic_optimizer.param_groups[0]['lr'])

            # Reset the environment
            state, _ = env.reset(seed=env_seed)
            env_seed += 1
            done = False
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

        # Evaluate the episodes
        if (t + 1) % eval_freq == 0:
            evaluations.append(evaluate_policy(policy, env_seed, env_name))
            np.save(f"./results/{filename}", evaluations)
            np.save(f"./results/{filename}_training_rewards", training_rewards)
            if save: policy.save(f"./models/{filename}")

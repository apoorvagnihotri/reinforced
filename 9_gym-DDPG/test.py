import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab as plt

import DDPG
import torch

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def run(env, agent, n_episodes=100, noise=0):
    rewards = []
    observations = []
    actions = []
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(2000):
            action = agent.act(state, noise)
            state, reward, done, _trunc, _info = env.step(action)
            observations.append(state)
            actions.append(action)
            ep_reward += reward
            if done or _trunc:
                break
        rewards.append(ep_reward)
        ep_reward = 0
    print(f'Mean reward: {np.mean(rewards)}')
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    return observations, actions, rewards

env_name = "Pendulum-v1"
eps=0.1
ts=32
lr=0.0001
s=None

with open(f"./results/DDPG_{env_name}-eps{eps}-t{ts}-l{lr}-s{s}-stat.pkl", 'rb') as f:
    data = pickle.load(f)
    rewards = np.asarray(data["rewards"])
    losses = np.array([(loss[0].item(), loss[1]) for loss in data["losses"]])

fig=plt.figure(figsize=(6,3.8))
plt.plot(running_mean(losses[:,0],10),label=f"Q loss")
plt.plot(running_mean(losses[:,1],10),label=f"pi loss")
plt.legend()

fig = plt.figure(figsize=(6, 3.8))
plt.plot(running_mean(rewards, 10), label=f"Rewards (eps={eps}, lr={lr}, ts={ts})")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.show()

def plot_Q_function(q_function, observations, actions, plot_dim1=0, plot_dim2=2,
                    label_dim1="cos(angle)", label_dim2="angular velocity"):
    plt.rcParams.update({'font.size': 12})
    values = q_function.predict(np.hstack([observations, actions]))

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot()
    surf = ax.scatter(observations[:, plot_dim1], observations[:, plot_dim2], c=values, cmap=cm.coolwarm)
    ax.set_xlabel(label_dim1)
    ax.set_ylabel(label_dim2)

    return fig


env = gym.make(env_name)
episodes=2000
eps=0.1
ts=32
lr=0.0001
checkpoint = f"./results/DDPG_{env_name}_{episodes}-eps{eps}-t{ts}-l{lr}-sNone.pth"

# Initialize the agent
agent = DDPG.DDPGAgent(env.observation_space, env.action_space)

# Load the checkpoint
checkpoint_data = torch.load(checkpoint)
agent.restore_state(checkpoint_data)
# Run 100 episodes with noise 0.2
observations, actions, rewards = run(env, agent, n_episodes=100, noise=0.2)
# Plot the Q-function
label_dim1, label_dim2 = "cos(angle)", "angular velocity"
fig = plot_Q_function(agent.Q, observations, actions,
                      plot_dim1=0, plot_dim2=2,
                      label_dim1=label_dim1, label_dim2=label_dim2)

plt.show()

## TEST LR AND UPDATE COMBINATIONS
update_frequencies = [20, 100]
learning_rates = [0.001, 0.0005, 0.0001, 0.00005]


# Function to load the agent and evaluate performance
def evaluate_model(checkpoint_path, agent, env, max_timesteps=1000):
    # Load the checkpoint
    saved_state = torch.load(checkpoint_path)
    agent.restore_state(saved_state)

    # Evaluate the model for a few episodes
    total_rewards = []
    for episode in range(1000):  # 1000 test episodes
        state, _info = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            action = agent.act(state, eps=0)  # No exploration during testing
            state, reward, done, truncated, _info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        total_rewards.append(total_reward)

    return np.mean(total_rewards)  # Average reward over test episodes


# Store results for each combination of update frequency and learning rate
results = {}

# Loop over all combinations of update frequency and learning rate
for update_freq in update_frequencies:
    for lr in learning_rates:
        # Construct the checkpoint filename based on the update frequency and learning rate
        checkpoint_path = f"./results/DDPG_Pendulum-v1_eps0.1_t32_l{lr}_update{update_freq}_2000.pth"
        print(f"Evaluating model: {checkpoint_path}")

        # Initialize the agent with the given learning rate and update frequency
        ddpg = DDPG.DDPGAgent(env.observation_space, env.action_space, actor_lr=lr, update_freq=update_freq)

        # Evaluate the model
        avg_reward = evaluate_model(checkpoint_path, ddpg, env)
        results[(update_freq, lr)] = avg_reward
        print(f"Avg reward for update_freq={update_freq}, learning_rate={lr}: {avg_reward}")

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the rewards for each update frequency
for update_freq in update_frequencies:
    rewards = [results[(update_freq, lr)] for lr in learning_rates]
    ax.plot(learning_rates, rewards, label=f"Update Frequency {update_freq}", marker='o')

ax.set_xscale('log')  # Log scale for learning rate
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Average Reward (last 5 episodes)")
ax.legend()
plt.title("Performance of Different Actor Learning Rates and Update Frequencies")
plt.show()



## HALFCHEETAH-v4
# Load the trained policy
policy_path = "./results/DDPG_HalfCheetah-v4_6000-eps0.1-t32-l0.0001-s1.pth"


# Create the environment and initialize the agent
env = gym.make("HalfCheetah-v4", render_mode="human")
ddpg = DDPG.DDPGAgent(env.observation_space, env.action_space)

# Load the saved policy state
saved_state = torch.load(policy_path)
ddpg.restore_state(saved_state)

# Test the policy in the environment
def test_policy(agent, env, max_timesteps=1000):
    observation, _ = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        env.render()
        action = agent.act(observation, eps=0)  # Deterministic action
        observation, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"Total Reward = {total_reward}")

# Run the simulation
test_policy(ddpg, env, max_timesteps=1000)
env.close()
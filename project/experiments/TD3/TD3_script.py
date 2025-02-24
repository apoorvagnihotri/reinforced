import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import copy

device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        # Actor NN
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        action = F.relu(self.layer1(state))
        action = F.relu(self.layer2(action))
        action = torch.tanh(self.layer3(action)) * self.max_action
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 NN
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

        # Q2 NN
        self.layer4 = nn.Linear(state_dim + action_dim, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1(state_action))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        q2 = F.relu(self.layer4(state_action))
        q2 = F.relu(self.layer5(q2))
        q2 = self.layer6(q2)
        return q1, q2

    def Q1(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1(state_action))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        return q1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, learning_rate):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Target policy smoothing is scaled with respect to the action scale
        self.policy_noise = 0.2 * self.max_action
        self.noise_clip = 0.5 * self.max_action

        # Define the needed TD3 parameters
        self.tau = 0.005
        self.delay_counter = 0
        self.delay_freq = 2
        self.discount = 0.99
        self.actor_lr = learning_rate
        self.critic_lr = learning_rate

        # Define the actor
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        # Define the critic
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), device=device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        self.delay_counter += 1
        with torch.no_grad():
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size=256)

            # Define the clipped noise
            target_action_noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # Select the next action based on the policy
            next_action = (self.actor_target(next_state) + target_action_noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get the current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute the critic loss and optimize it
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.delay_counter % self.delay_freq == 0:
            # Compute the actor loss and optimize it
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target models
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_optimizer_lr(self, new_actor_lr, new_critic_lr):
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_actor_lr

        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_critic_lr

        print(f"Actor optimizer updated with new learning rate: {new_actor_lr}")
        print(f"Critic optimizer updated with new learning rate: {new_critic_lr}")

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class PinkNoise:
    def __init__(self, action_dim, beta=0.95):
        self.action_dim = action_dim
        self.beta = beta  # Defines the noise color (1.0 is standard pink noise, we use 0.95 for proper noise decay)
        self.pinknoise = np.zeros(action_dim)

    def reset(self):
        self.pinknoise = np.zeros(self.action_dim)

    def sample(self):
        whitenoise = np.random.normal(0, 1, self.action_dim)
        self.pinknoise = self.beta * self.pinknoise + (1 - self.beta) * whitenoise
        return self.pinknoise

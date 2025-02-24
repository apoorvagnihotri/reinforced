import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal



class PolicyNet(nn.Module):
    def __init__(self, learning_rate, state_dim, action_dim, config):
        super(PolicyNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        ).to(config.device)
        
        self.mean_branch = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(config.device)
        
        self.std_branch = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(config.device)

        self.config = config
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(config.init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_alpha)

    def forward(self, state):
        state = state.to(self.config.device)
        features = F.relu(self.feature_extractor(state))
        
        mu = self.mean_branch(features)
        sigma = F.softplus(self.std_branch(features))
        sigma = torch.clamp(sigma, min=1e-6)
        
        dist = Normal(mu, sigma)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample)
        
        # Apply tanh transformation to bound actions
        action = torch.tanh(sample)
        # Adjust log probability for tanh transformation
        adjusted_log_prob = log_prob - torch.log(1 - torch.tanh(sample).pow(2) + 1e-7)
        return action, adjusted_log_prob

    def train_net(self, q1, q2, batch):
        s, _, _, _, _ = batch
        sampled_action, log_prob = self.forward(s)
        total_log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = -self.log_alpha.exp() * total_log_prob #Entropy loss term, weighted by alpha

        q1_val, q2_val = q1(s,sampled_action), q2(s,sampled_action)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        
        
        policy_loss = -min_q - entropy # for gradient ascent 
        self.optimizer.zero_grad()
        policy_loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (total_log_prob + self.config.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return min_q, total_log_prob
        

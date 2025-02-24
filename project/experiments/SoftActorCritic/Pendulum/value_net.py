import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config

class QNet(nn.Module):
    def __init__(self, lr, state_size, action_size):
        super().__init__()
        self.state_layer = nn.Linear(state_size, 128).to(config.device)
        self.action_layer = nn.Linear(action_size, 128).to(config.device)
        self.combined_layer = nn.Sequential(
            nn.Linear(256, 128).to(config.device),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(128, 1).to(config.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        state = state.to(config.device)
        action = action.to(config.device)
        state_out = F.relu(self.state_layer(state))
        action_out = F.relu(self.action_layer(action))
        merged = torch.cat([state_out, action_out], dim=1)
        combined = F.relu(self.combined_layer(merged))
        q_value = self.output_layer(combined)
        return q_value

    def train_net(self, target, batch):
        s, a, r, s_prime, done = batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - config.tau) + param.data * config.tau)
            
def compute_target(policy_net, q1, q2, batch):
    s, a, r, s_prime, done = batch

    with torch.no_grad():
        a_prime, log_prob= policy_net(s_prime)
        # Sum log-probabilities across action dimensions
        reduced_log_prob = log_prob.sum(dim=-1, keepdim=True)  # Shape: [bs, 1]
        entropy = -policy_net.log_alpha.exp() * reduced_log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime) #q1_val.shape: [bs, 1]
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        #print(r.config.device, done.config.device, min_q.config.device, entropy.config.device)
        target = r + config.gamma * done * (min_q + entropy)

    return target
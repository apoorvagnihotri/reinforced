import numpy as np
import gymnasium as gym
from importlib import reload
import time
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


import gymnasium as gym
import pygame
import torch

import datetime
import math

import numpy as np
import collections, random


import os
import sys

import config

try:
    base_path = os.path.dirname(__file__)
except NameError:
    base_path = os.getcwd()

parent_dir = os.path.abspath(os.path.join(base_path, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from models.memory import ReplayBuffer
from models.policy_net import PolicyNet
from models.value_net import QNet, compute_target




np.set_printoptions(suppress=True)


log_dir = f"runs/HalfCheetah/TE_4_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
model_dir = os.path.join(log_dir, "models")
os.makedirs(model_dir, exist_ok=True)
writer = SummaryWriter(log_dir)  # Create a new logging directory


hyperparams = {
    "lr_pi": config.lr_pi,
    "lr_q": config.lr_q,
    "init_alpha": config.init_alpha,
    "gamma": config.gamma,
    "batch_size": config.batch_size,
    "buffer_limit": config.buffer_limit,
    "tau": config.tau,
    "target_entropy": config.target_entropy,
    "lr_alpha": config.lr_alpha,
    "total_eps": config.episodes,
}

hyperparam_text = "\n".join([f"{key}: {value}" for key, value in hyperparams.items()])

# Log all hyperparameters to TensorBoard
writer.add_text("Hyperparameters", hyperparam_text)

def main(score_list):
    env = gym.make('HalfCheetah-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f'state_dim: {state_dim},  action_dim: {action_dim}')
        
    
    memory = ReplayBuffer(config)
    q1, q2, q1_target, q2_target = QNet(config.lr_q, state_dim, action_dim, config).to(config.device), QNet(config.lr_q, state_dim, action_dim, config).to(config.device), QNet(config.lr_q, state_dim, action_dim, config).to(config.device), QNet(config.lr_q, state_dim, action_dim, config).to(config.device)
    pi = PolicyNet(config.lr_pi, state_dim, action_dim, config).to(config.device)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 20
    

    for n_epi in range(config.episodes):
        s, _ = env.reset()
        done = False
        count = 0            

        while count < 2000 and not done:


            a, log_prob= pi(torch.from_numpy(s).float())
            a = a.cpu().detach().numpy()
                
            s_prime, r, done, truncated, info = env.step(a)

            memory.put((s, a, r/10.0, s_prime, done))
            score +=r
            s = s_prime
            count += 1

        sac_losses = []
        score_list.append(score)
        writer.add_scalar("Training/Eps_Score", score, n_epi)
        score = 0.0

        if memory.size()>4000:
            for i in range(50):
                mini_batch = memory.sample(config.batch_size)
                mini_batch = tuple(t.to(config.device) for t in mini_batch)

                td_target = compute_target(pi, q1_target, q2_target, mini_batch, config)


                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)

                #Update policy net
                sac_loss = pi.train_net(q1, q2, mini_batch)  #sac_loss.shape : [32,1] 
                sac_losses.append(sac_loss)


                q1.soft_update(q1_target)
                q2.soft_update(q2_target)





        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = sum(score_list[-print_interval:]) / print_interval
            min_q_values = [item[0].mean().item() for item in sac_losses]
            entropy_values = [item[1].mean().item() for item in sac_losses]

            # Compute overall means using numpy
            mean_min_q = np.mean(min_q_values)
            mean_entropy = np.mean(entropy_values)

            # Print results
            print(f"# Episode: {n_epi}, Avg Score: {avg_score:.1f}, Alpha: {pi.log_alpha.exp():.4f}")
            print(f"Mean min_q: {mean_min_q:.4f}, Mean entropy: {mean_entropy:.4f}, alpha*log_prob: {mean_entropy*pi.log_alpha.exp():.4f}")
            
            writer.add_scalar("Training/Avg_Score", avg_score, n_epi)
            writer.add_scalar("Training/Avg_Min_Q", mean_min_q, n_epi)
            writer.add_scalar("Training/Avg_Entropy", mean_entropy, n_epi)
            writer.add_scalar("Training/A*logP", mean_entropy*pi.log_alpha.exp(), n_epi)
            
            


        if n_epi% 100 == 0 or n_epi == config.episodes-1:
            model_path = os.path.join(model_dir, f"policy_net.pth")
            torch.save(pi.state_dict(), model_path)
        


    env.close()
    

score_list = []
if __name__ == '__main__':
    main(score_list)
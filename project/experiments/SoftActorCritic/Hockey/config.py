import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#SAC Hyperparameters
lr_pi           = 3e-4
lr_q            = 3e-4
init_alpha      = 0.01
gamma           = 0.99
batch_size      = 256
buffer_limit    = 50000
tau             = 0.005 # for target network soft update
target_entropy  = -4.0 
lr_alpha        = 0.0003  # for automated alpha update
episodes        = 15000
weight_touch_puck = 1.0
init_time_steps = 25
max_timesteps = 250
increase_every = 2500
increase_rate = 25
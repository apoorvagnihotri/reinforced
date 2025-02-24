import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#SAC Hyperparameters
lr_pi           = 3e-4
lr_q            = 3e-4
init_alpha      = 0.01
gamma           = 0.99
batch_size      = 256
buffer_limit    = 100000
tau             = 0.005 # for target network soft update
target_entropy  = -6.0 
lr_alpha        = 0.0003  # for automated alpha update
episodes        = 10000
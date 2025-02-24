import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

lr_pi           = 3e-4
lr_q            = 3e-4
init_alpha      = 0.01
gamma           = 0.99
batch_size      = 64
buffer_limit    = 50000
tau             = 0.005 
target_entropy  = -1.0 
lr_alpha        = 0.0003
episodes        = 5000

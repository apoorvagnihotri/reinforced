## Soft Actor-Critic Implementation

This folder contains all scripts, notbooks and saved model parameters for the Soft Actor-Critic Algorithm. SAC is trained on Pendulum, HalfCheetah and Hockey environments.


1) Create virtual environment using the yml file.

2) The hyperparameters for each gym environment are in the respective folder/config.py file.

3) To start training, simply run SAC.py

Tensorboard is used to log all training statistics and save model parameters. Generated plots are in plots.ipynb notebook.
To simulate the trained hockey agent, look at Hockey/play_hockey.ipynb

## TD3 Implementation
This folder contains the TD3 and main scripts, as well as the model checkpoints and training/evaluation rewards for each environment
(HalfCheetah-v4, Pendulum-v1, Hockey weak bot and Hockey strong bot). The figures used for the report were obtained by plotting the training and evaluation rewards
of the code with and without modifications using the visualize_results script:
1) HalfCheetah-v4: seed 100 is for no modifications, seed 104 is for modifications.
2) Pendulum-v1: seed 101 is for no modifications, seed 105 is for modifications.
3) Hockey-weak_bot: seed 25 is for no modifications, seed 23 is for modifications.
4) Hockey-strong_bot: seed 26 is for no modifications, seed 24 is for modifications.

To run the TD3 code execute the main_script.py file, this will train the agent using TD3 with modifications. The environment, seed and other parameters can be
changed inside the main_script.py file. Right now it is set for the Hockey environment with strong bot.

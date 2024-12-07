import gymnasium as gym
from PPO import PPO, Memory
from PIL import Image
import torch
import optparse

def test():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="LunarLander-v2",
                         help='Environment (default %default)')
    optParser.add_option('-r', '--render',action='store_true',
                         dest='render',
                         help='render Environment if given')
    optParser.add_option('-g', '--gif',action='store_true',
                         dest='gif',
                         help='render Environment into animaged gif')
    optParser.add_option('-f', '--file',action='store', type='string',
                         dest='filename',
                         help='filename of checkout')

    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    render_mode = "human" if opts.render else None
    save_gif = opts.gif
    if save_gif:
        if render_mode is not None:
            print("overwrite render-mode to image")
        render_mode = "rgb_array"
        import os
        os.makedirs('./gif', exist_ok=True)
        print("to get a gif for episode 1 run \"convert -delay 1x30 ./gif/01_* ep01.gif\"")
    env = gym.make(env_name, render_mode = render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_timesteps = 300
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 10
    max_timesteps = 300

    filename = opts.filename

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    ppo.policy_old.load_state_dict(torch.load(filename))
    rewards = []

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _trunc, _info = env.step(action)
            ep_reward += reward
            if save_gif:
                 img = env.render()
                 img = Image.fromarray(img)
                 img.save(f'./gif/{ep:02}-{t:03}.jpg')
            if done:
                break
        rewards.append(ep_reward)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0

    env.close()


if __name__ == '__main__':
    test()

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

import memory as mem
from feedforward import Feedforward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100], learning_rate = 0.0002):
        # TODO: Setup network with right input and output size (using super().__init__)
        ##########
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, output_size=action_dim)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        ##########
        # END
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        # TODO: implement the forward pass.
        ##########
        # Get all Q-values for each state
        q_values = self.forward(observations)
        return q_values
        ##########

# Ornstein Uhlbeck noise, Nothing to be done here
class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)

class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "update_target_every": 100,
            "use_target_net": True
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']

        self.action_noise = OUNoise((self._action_n))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes=self._config["hidden_sizes_critic"],
                           learning_rate=self._config["learning_rate_critic"])
        # target Q Network
        self.Q_target = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes=self._config["hidden_sizes_critic"],
                                  learning_rate=0)

        high, low = torch.from_numpy(self._action_space.high), torch.from_numpy(self._action_space.low)
        # TODO:
        # The activation function of the policy should limit the output the action space
        # and makes sure the derivative goes to zero at the boundaries
        # Use Tanh, which is between -1 and 1 and scale it to [low, high]
        # Hint: use torch.nn.Tanh()(x)
        ##########
        output_activation = lambda x: (torch.nn.Tanh()(x) * (high - low) / 2) + (high + low) / 2
        ##########

        self.policy = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = output_activation)
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = output_activation)

        self._copy_nets()

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        # TODO: implement this: use self.action_noise() (which provides normal noise with standard variance)
        ##########
        """Compute action using policy and add noise for exploration."""
        if eps is None:
            eps = self._eps
        obs_tensor = torch.from_numpy(observation.astype(np.float32))
        with torch.no_grad():
            action = self.policy(obs_tensor).numpy()
        noise = self.action_noise() * eps
        return np.clip(action + noise, self._action_space.low, self._action_space.high)
        ##########

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()

        for i in range(iter_fit):
            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:,0])) # s_t
            a = to_torch(np.stack(data[:,1])) # a_t
            rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:,3])) # s_t+1
            done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)
            # TODO: Implement the rest of the algorithm

            # assign q_loss_value  and actor_loss to we stored in the statistics
            ##########
            # Critic loss (TD error)
            with torch.no_grad():
                next_action = self.policy_target(s_prime)
                q_target_next = self.Q_target(s_prime)
                y = rew + self._config["discount"] * (1 - done) * q_target_next

            q_value = self.Q.Q_value(s, a)
            q_loss_value = torch.nn.functional.mse_loss(q_value, y)

            self.Q.optimizer.zero_grad()
            q_loss_value.backward()
            self.Q.optimizer.step()

            # Actor loss
            pred_action = self.policy(s)
            actor_loss = -self.Q.Q_value(s, pred_action).mean()

            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            ##########

            losses.append((q_loss_value , actor_loss.item()))

        return losses


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=1,
                         help='random seed (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy
    random_seed = opts.seed
    #############################################


    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ddpg = DDPGAgent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                     update_target_every = opts.update_every)

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/DDPG_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, max_episodes+1):
        ob, _info = env.reset()
        ddpg.reset()
        total_reward=0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = ddpg.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward+= reward
            ddpg.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            if done or trunc: break

        losses.extend(ddpg.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(ddpg.state(), f'./results/DDPG_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()

if __name__ == '__main__':
    main()


# def main(lr, update_every, random_seed):
#     optParser = optparse.OptionParser()
#     optParser.add_option('-e', '--env', action='store', type='string',
#                          dest='env_name', default="Pendulum-v1",
#                          help='Environment (default %default)')
#     optParser.add_option('-n', '--eps', action='store', type='float',
#                          dest='eps', default=0.1,
#                          help='Policy noise (default %default)')
#     optParser.add_option('-t', '--train', action='store', type='int',
#                          dest='train', default=32,
#                          help='number of training batches per episode (default %default)')
#     optParser.add_option('-l', '--lr', action='store', type='float',
#                          dest='lr', default=lr,
#                          help='learning rate for actor/policy (default %default)')
#     optParser.add_option('-m', '--maxepisodes', action='store', type='float',
#                          dest='max_episodes', default=2000,
#                          help='number of episodes (default %default)')
#     optParser.add_option('-u', '--update', action='store', type='float',
#                          dest='update_every', default=update_every,
#                          help='number of episodes between target network updates (default %default)')
#     optParser.add_option('-s', '--seed', action='store', type='int',
#                          dest='seed', default=random_seed,
#                          help='random seed (default %default)')
#     opts, args = optParser.parse_args()
#
#     # Create the environment
#     env_name = opts.env_name
#     env = gym.make(env_name)
#
#     # Set the seed for random number generators
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#         np.random.seed(random_seed)
#
#     ddpg = DDPGAgent(env.observation_space, env.action_space, eps=opts.eps, learning_rate_actor=opts.lr,
#                      update_target_every=opts.update_every)
#
#     rewards = []
#     lengths = []
#     losses = []
#     timestep = 0
#
#     def save_statistics():
#         file_name = f"./results/DDPG_{env_name}_eps{opts.eps}_t{opts.train}_l{opts.lr}_update{opts.update_every}_s{random_seed}_stat.pkl"
#         with open(file_name, 'wb') as f:
#             pickle.dump({"rewards": rewards, "lengths": lengths, "eps": opts.eps, "train": opts.train,
#                          "lr": opts.lr, "update_every": opts.update_every, "losses": losses}, f)
#
#     # Training loop
#     for i_episode in range(1, opts.max_episodes + 1):
#         ob, _info = env.reset()
#         ddpg.reset()
#         total_reward = 0
#         for t in range(2000):  # max_timesteps
#             timestep += 1
#             done = False
#             a = ddpg.act(ob)
#             ob_new, reward, done, trunc, _info = env.step(a)
#             total_reward += reward
#             ddpg.store_transition((ob, a, reward, ob_new, done))
#             ob = ob_new
#             if done or trunc: break
#
#         losses.extend(ddpg.train(opts.train))
#
#         rewards.append(total_reward)
#         lengths.append(t)
#
#         # Save checkpoint every 500 episodes
#         if i_episode % 500 == 0:
#             checkpoint_filename = f'./results/DDPG_{env_name}_eps{opts.eps}_t{opts.train}_l{opts.lr}_update{opts.update_every}_s{random_seed}_{i_episode}.pth'
#             torch.save(ddpg.state(), checkpoint_filename)
#             save_statistics()
#
#         # Logging
#         if i_episode % 20 == 0:  # log_interval
#             avg_reward = np.mean(rewards[-20:])
#             avg_length = int(np.mean(lengths[-20:]))
#             print(f'Episode {i_episode} \t avg length: {avg_length} \t reward: {avg_reward}')
#
#     save_statistics()


# if __name__ == '__main__':
#     learning_rates = [0.001, 0.0005, 0.0001, 0.00005]
#     update_frequencies = [20, 100]
#
#     for lr in learning_rates:
#         for update in update_frequencies:
#             # Generate a unique seed based on the combination of learning rate and update frequency
#             seed = hash((lr,update)) % 1000  # Limit the seed to a 3-digit integer (0-999)
#             print(f"Running experiment with lr={lr}, update_every={update}, seed={seed}")
#             main(lr=lr, update_every=update, random_seed=seed)
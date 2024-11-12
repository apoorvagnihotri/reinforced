# Homework 4

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)

## Question 1: Q-Learning and SARSA

### Part a)

Q-Learning is considered an off-policy control method because in the learning stage it doesn't use current policy to update the Q-value for a (state, action) pair but instead chooses the maximum expected future reward for the next state s' over all possible actions. 

### Part b)

No, Q-learning and SARSA are not the same algorithm even if action selection is greedy. Q-learning is off-policy and SARSA is on-policy, so SARSA takes into account the policy. This may result into SARSA producing a different update than Q-learning in cases of ties in Q-values due to a tie-breaking rule or randomness in the policy.

### Part c)

a) The action taken should be the left path, as the expected return from that action is 0.1 due to the reward being drawn from a normal distribution with mean 0.1, which is greater than the reward of 0 from the right path.

b) Q-learning tries to maximize the Q-value of the actions, so over time it will favor the left path over the right path due to the expected return being 0.1 vs 0. However, there is a high variance in the reward from the left path, so the Q-value for that path will have high variance as well, making it fluctuate along the episodes until it eventually converges to the expected Q-value of 0.1 for the left action, which takes preference over the expected Q-value of 0 for the right path.

## Question 2.1: Q-Learning

### Part a)

The values learned by Q-learning and value iteration are different due to the following possible reasons:

1.  Q-learning is a model free method, it learns from the interactions with the environment rather than a complete model of state transitions and rewards. It relies on exploration to gradually improve its Q-values, which can lead to variations in the learned values when running for a limited number of episodes.

1. Since Q-learning doesn't have access to the full model, it may not explore all states and actions sufficiently in 100 episodes. On the other hand, value iteration uses the full environment model to compute values directly. It converges to the true optimal values because it evaluates all possible actions in every state based on known probabilities. 

To make the values from Q-learning closer to the optimal values we could increase the number of episodes and/or increase epsilon so that the agent takes random actions more frequently, leading to more exploration:

1. Increasing the number of episodes to 300 and keeping the same value of epsilon=0.2, the agent learns the values of the states which are much closer to the ones obtained from value iteration. 

1. Keeping the number of episodes to 100 and increasing epsion to 0.4 leads to more random actions and more exploration. This also brings the values closer to those obtained from value iteration. 

### Part b)

Q-Learning agent is unable to learn the optimal policy unlike value iteration in case of BridgeGrid due to the following reasons:

1. Q-Learning relies on exploration to find paths to high-reward states, but with such high penalties nearby, the agent avoids exploring far to the right due to the risk of ending up in a high-penalty state.

1. The high penalty states cause to agent to learn to avoid exploring those states, and alwyas picks the action giving the highest Q-value(exploitation). Consequently it avoids exploring towards the rightmost high reward state.

1. Even running for a lot of episodes and higher value of epsilon, the agent still avoids the path on the right and chooses the safe action. 

### Part c)

In our case of training the Q-learning agent on CliffGrid for 300 episodes, we obtain an average return of -25.259 from the start state. Meanwhile, the value of the start after training is -1.68. They are so different due to the following reasons:

1. The Q-value learned by Q-learning for the start state represents the agent’s best estimate of the optimal return it can achieve from that state by following the learned policy. Q-learning learns the estimates based on the maximum possible Q-value for future states. This results in a value that reflects the best possible performance if the agent follows the optimal path, avoiding penalties as much as possible.

1. In contrast, the average return during training episodes includes all exploration steps and mistakes made by the agent, especially early in training when the agent is exploring the environment. These returns are affected by random exploration steps and suboptimal actions that lead the agent into high-penalty states (-100 reward). 

1. By the time the agent learns to avoid the penalty states more effectively, many episodes have already contributed large negative returns to the average. Even if the agent improves it's policy, it still occasionally explores due to eplsion-greedy exploration, visiting penalty states and consequently keeping the average return lower than the optimal policy's expected return.

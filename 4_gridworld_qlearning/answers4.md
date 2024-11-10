# Homework 4

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)

## Question 1: Q-Learning and SARSA

### Part a)

Q-Learning is considered an off-ploicy control method because in the learning stage it doesn't use current policy to update the Q-value for a (state, action) pair but instead chooses the maximum expected future reward for the next state s' over all possible actions. 

### Part b)



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

# Homework 2 - Question 1: State-Action Value Function and Policy Iteration

## Part a) Gridworld Q-Values

To find the Q-values q^π(s,a) for the given states and actions under the equiprobable random policy, we need to consider:
- Each action has probability 1/4 under the random policy
- Reward is -1 for all transitions
- The given value function can be used to determine future state values

| 0   | -14 | -20 | -22 |
| -14 | -18 | -20 | -20 |
| -20 | -20 | -18 | -14 |
| -22 | -20 | -14 | 0   |


Reeterating the states
| 0   | 1   | 2   | 3   |
| 4   | 5   | 6   | 7   |
| 8   | 9   | 10  | 11  |
| 12  | 13  | 14  | 15  |

For each state-action pair:

### q^π(11, down):
- Moving down from state 11 leads to state 15 (terminal state)
- Terminal state has value 0
- Q-value = Immediate reward + Value of next state
- q^π(11, down) = -1 + 0 = -1

### q^π(7, down):
- Moving down from state 7 leads to state 11
- State 11 has value -14 according to the given value function
- q^π(7, down) = -1 + (-14) = -15

### q^π(9, left):
- Moving left from state 9 leads to state 8
- State 8 has value -16 according to the given value function
- q^π(9, left) = -1 + (-16) = -17

## Part b) Optimal Value Function

The optimal value function v*(s) in terms of q*(s,a) is:

v*(s) = max_a q*(s,a)

This equation states that the optimal value of a state is equal to the maximum Q-value over all possible actions in that state.

## Part c) Optimal Q-Function

The optimal Q-function q*(s,a) in terms of v*, P, and R is:

q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) v*(s')

Where:
- R(s,a) is the reward function
- γ is the discount factor
- P(s'|s,a) is the transition probability
- v*(s') is the optimal value function of the next state

## Part d) Optimal Policy

The optimal policy π* in terms of q* is:

π*(a|s) = 1 if a = argmax_a q*(s,a)
π*(a|s) = 0 otherwise

This defines a deterministic policy that always selects the action with the highest Q-value in each state.

## Part e) Bellman Expectation Equation for Q-values

The Bellman Expectation Equation for q^π(s,a) in terms of only q values is:

q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') q^π(s',a')

This equation expresses the Q-value of a state-action pair in terms of:
- The immediate reward R(s,a)
- The discounted future Q-values of all possible next states s' and actions a'
- Weighted by their transition probabilities P(s'|s,a)
- And the policy probabilities π(a'|s')

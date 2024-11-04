# Homework 3

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)

## Question 1: Recap

### Part a) What is an Markov reward process (MRP)?

A Markov Reward Process (MRP) is a mathematical framework used in the study of reinforcement learning and stochastic processes. It extends the concept of a Markov Chain by incorporating rewards, allowing for the evaluation of policies based on the expected returns. Here are the key components of an MRP:

-  States (S): A finite or infinite set of states that the process can be in. Each state represents a situation in which an agent can find itself.

-  Transition Probabilities (P): A matrix 𝑃(𝑠′∣𝑠) that defines the probability of transitioning from one state 𝑠 to another state 𝑠′. This reflects the dynamics of the system and must satisfy the Markov property, meaning that the future state depends only on the current state and not on the history of past states.

-  Rewards (R): A reward function 𝑅(𝑠) that assigns a numerical value (reward) to each state 𝑠. This indicates the immediate reward received upon entering that state.

-  Discount Factor (γ): A value 𝛾 (0 ≤ γ < 1) that represents the importance of future rewards compared to immediate rewards. It is used to compute the expected return over time.

Markov Reward Processes are foundational in reinforcement learning, as they help to model environments where an agent interacts with the world and learns to optimize its behavior based on the rewards received over time. They form the basis for more complex structures like Markov Decision Processes (MDPs), which include actions that influence state transitions.

### Part b) How can I reduce a Markov Decision Process (MDP) to a MRP?

Reducing a Markov Decision Process (MDP) to a Markov Reward Process (MRP) involves removing the aspect of decision-making by fixing a specific policy for the agent to follow. Here’s how this reduction works:

1. Define a Policy 𝜋
In an MDP, an agent selects actions based on a policy 𝜋(𝑎∣𝑠), which gives the probability of taking action
𝑎 in state 𝑠. To convert an MDP to an MRP, choose a specific policy 𝜋 for the agent, effectively removing the decision-making element.

2. Construct State Transition Probabilities under the Policy
In an MDP, the state transitions depend on both the current state and the chosen action. By fixing a policy, you can calculate a new set of state transition probabilities based solely on the states:

$$P^𝜋(s′∣s)=∑_𝑎[𝜋(a∣s)𝑃(s′∣s,a)]$$

Here, 𝑃^𝜋(𝑠′∣𝑠) is the probability of transitioning from state 𝑠 to 𝑠′ under the policy 𝜋. It’s the weighted sum of the transition probabilities 𝑃(𝑠′∣𝑠,𝑎) across all possible actions, weighted by the probability 𝜋(𝑎∣𝑠) of selecting each action.

3. Compute the Expected Reward per State
In an MDP, the reward depends on both the state and the action taken. To reduce this to an MRP, calculate the expected reward for each state under the chosen policy:

R^\pi(s)=∑_a\pi(a∣s)R(s,a)

4. Resulting Markov Reward Process (MRP)
The MDP is now reduced to an MRP with:

States 
$S$: The same set of states as in the MDP.
Transition Matrix $P^π$ : The state-to-state transition probabilities under the fixed policy $\pi$.
Reward Function $R^\pi$ : The expected reward function for each state under $\pi$.
Discount Factor γ: The same discount factor from the original MDP.

By following these steps, the MDP is reduced to a Markov Reward Process, which removes the need for actions and decision-making, allowing us to evaluate the policy's performance solely based on the states, transitions, and rewards.



## Question 3: Policy Iteration
### Part a) Create a Policy Iteration Agent in agent.py
### Part b) 

When we use the default noise, we need a total of 2 iterations to allow the start state to have a non-zero value. When we turn of the noise in the transitions probabilities, policy iteration takes a lot longer to get the start state to have more than 0 as state value. In our case we observed it took 6 iterations for start state to have non-zero value.

### Part c) 

It takes a total of 10 policy iterations for the algorithm to converge both in case of default noise and no-noise.

### Part d)
Advantages of Policy Iteration

* Often converges in fewer iterations than value iteration
* Can be more efficient in environments where good policies are easlier to find than optimal values.

Disadvantages of Policy Iteration

* Each iteration is more computationally expensive due to the policy evalution step
* Requires storing an explicit policy for each state, which can be a problem if the state space is too large.
* Policy evalution step may need multiple iterations to converge.


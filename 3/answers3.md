# Homework 3

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)

## Question 1: Recap

### Part a)

A Markov reward process is a modified Markov chain which contains an added term that represents the reward rate associated to each state transition in the process.

### Part b)

A Markov Decision Process can be reduced to a Markov Reward Process by removing the decision-making aspect. This can be done by fixing a deterministic policy that the agent must follow instead of letting the agent choose a decision based on probability.

### Part c)

MRPs can be solved in closed form because they have no decision-making, only states, transitions and rewards. This makes it possible to describe the system as a set of linear equations, which have a closed form. MDPs include actions, so the agent must make decisions that influence the outcomes, introducing complexity to the problem. To find the optimal solution, we require a non-linear and iterative approach, which doesn't have a closed form.

## Question 2: TD($\lambda$) updates
### Part a)
$G_t^{(n)}$ is defined as:

$$\left(
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
\right)$$

Then, from this formula we derive $G_t^{(n-1)}$:

$$\left(
G_t^{(n-1)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{n-2} R_{t+n-1} + \gamma^{n-1} V(S_{t+n-1})
\right)$$

Finally, we derive $G_{t+1}^{(n-1)}$:

$$\left(
G_{t+1}^{(n-1)} = R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \dots + \gamma^{n-2} R_{t+n} + \gamma^{n-1} V(S_{t+n})
\right)$$

Comparing the last equation with the $G_t^{(n)}$ definition, we can see that:

$$\left(
G_{t+1}^{(n-1)} = \frac{G_t^{(n)} - R_{t+1}}{\gamma}
\right)$$

Therefore, the analogous recursive relationship for n-step return $G_t^{(n)}$ is:

$$\left(
G_t^{(n)} = R_{t+1} + \gamma G_{t+1}^{(n-1)}
\right)$$


### Part b)
$G_t^{(\lambda)}$ is defined as:

$$\left(
G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
\right)$$

Substituting the expression found in the previous part:

$$\left(
G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1}( R_{t+1} + \gamma G_{t+1}^{(n-1)})
\right)$$

Separating into two terms:

$$\left(
G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} R_{t+1} +  (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \gamma G_{t+1}^{(n-1)}
\right)$$

For the first term, $R_{t+1}$ is multiplying so we can take it out of the sum to get:

$$\left(
(1 - \lambda)  R_{t+1}  \sum_{n=1}^{\infty} \lambda^{n-1} = \frac{(1 - \lambda)  R_{t+1}}{(1 - \lambda)} = R_{t+1}
\right)$$

For the second term, we expand the first term of the sum:

$$\left(
(1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \gamma G_{t+1}^{(n-1)} = \gamma ((1 - \lambda)V(S_{t+1}) + (1 - \lambda) \sum_{n=2}^{\infty} \lambda^{n-1} G_{t+1}^{(n-1)})
\right)$$

$$\left(
(1 - \lambda) \sum_{n=2}^{\infty} \lambda^{n-1} G_{t+1}^{(n-1)} = \lambda (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t+1}^{(n)} = \lambda G_{t+1}^{\lambda}
\right)$$

Plugging these two simplifications into the original formula:

$$\left(
G_t^{\lambda} = R_{t+1} + \gamma ((1 - \lambda)V(S_{t+1}) + \lambda G_{t+1}^{\lambda})
\right)$$

## Question 3: Policy Iteration
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


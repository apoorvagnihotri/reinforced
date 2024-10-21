## Assignment 1: Gridworld Exploration
#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)
#### 1. Optimal Policy
Using the following formula for the return of this system we can solve this exercise: $G_t = R_{t+1} + \gamma\cdot R_{t+2} $, where $R_{t+1}$ = 1 for the left action and 0 for the right action, and $R_{t+2}$ = 0 for the left action and 2 for the right action.
Following are different cases with different values of $\gamma$ and the optimal policy for each case.
    
 * If we consider $\gamma = 0$, the optimal policy maximizes the immediate reward as $G_t = R_{t+1}$. In this case, the optimal policy is $\pi_{left}$. This is because the reward for moving to the left in the first state is 1. Meanwhile, the other action (right) leads to an immediate reward of 0. Thus, the optimal policy is to move to the left at each step.
        
 * If we consider $\gamma = 0.9$, $G_t = R_{t+1} + 0.9\cdot R_{t+2}$. The optimal policy is $\pi_{right}$. This is because the reward for moving to the right in the first time step is 0, but the reward in the next timestep is 2, which when multiplied by 0.9 (the discount factor) leads to a total reward of 0 + 1.8. Meanwhile, the other action (left) leads to an immediate reward of 1 and a delayed reward of 0 (multiplied by the discount of 0.9). $\pi_{left}$ gives a cumulative reward of 1. Thus, we choose the optimal policy as $\pi_{right}$.
    
 * If we consider $\gamma = 0.5$, either of the policies is optimal because in either case, we get the same reward $G_t = 1 + 0.5\cdot0 = 0 + 0.5\cdot2 = 1$. The optimal policy is $\pi_{left}$ or $\pi_{right}$.

#### 2. Value Estimation in Grid Worlds: implement return computation and value estimation
a. I tried the following values of k (num of episodes) to calculate the mean and standard deviation of the long-term average discounted rewards of the start state using MazeGrid environment if we were to use the random agent.

        | k     | mean     | std      |
        |-------|----------|----------|
        | 1     | 0.000001 | 0        |
        | 10    | 0.000010 | 0.000030 |
        | 100   | 0.000337 | 0.001832 |
        | 1000  | 0.002951 | 0.015572 |
        | 5000  | 0.002113 | 0.012364 |
        | 10000 | 0.002509 | 0.014818 |

  b. To get a 95% confidence that our mean is within $\pm 0.0004$ of the true mean, we use the law of large numbers. We can assume that the returns from different episodes are iid. According to CLT, we know that $n$ independent samples from a population with mean $\mu$ and standard deviation $\sigma$ will have the mean $X^{bar}$ distributed approximately with $N(\mu, \frac{\sigma}{\sqrt{n}})$. Now we use the formula that relates confidence intereval of the mean with standard deviation and the number of samples. This is a result of using the formula of a Guassian distribution.
    
$n = (z_{\frac{\alpha}{2}} * \sigma / E)^2$

Where:
z = 1.96 (for 95% confidence)
σ = standard deviation (from part a)
E = 0.0004 (desired margin of error)

We can only use the above formula when we already have a good estimate of the standard deviation. In this case, we can use the standard deviation from the previous part to calculate the number of samples needed to get a 95% confidence that our mean is within $\pm 0.0004$ of the true mean. Since we observed that the standard deviation kind of stopped changing after 1000 episodes, we can use the standard deviation from 10000 episodes to calculate the number of samples needed.

$n = (1.96 * 0.014818 / 0.0004)^2 = 5271.95$

This means we need around ~5000 episodes to get a 95% confidence that our mean is within $\pm 0.0004$ of the true mean.

  c. We here need a margin of error +- 0.05. Below is the table we calculated in part a for the new environment "DiscountGrid" with a discount factor of 0.95. 

        | k     | mean      | std      |
        |-------|-----------|----------|
        | 1     | +0.277390 | 0        |
        | 10    | -7.734860 | 3.094124 |
        | 100   | -6.884108 | 3.358412 |
        | 1000  | -6.336808 | 3.904625 |
        | 5000  | -6.485487 | 3.727223 |
        | 10000 | -6.311838 | 3.840347 |
    
We can use the same formula as in part b to calculate the number of samples needed to get a 95% confidence that our mean is within $\pm 0.05$ of the true mean. We can use the standard deviation from 10000 episodes to calculate the number of samples needed.

$n = (1.96 * 3.840347 / 0.05)^2 = 22662$

This means we need around ~22662 episodes to get a 95% confidence that our mean is within $\pm 0.05$ of the true mean. Since the suggested episodes are more than 10000, we recalculated the mean and the standard deviation of the long term discounted rewards for 30000 episodes. 

        | k     | mean      | std      |
        |-------|-----------|----------|
        | 30000 | -6.376765 | 3.792862 |

  d. Getting the mean long-term discounted rewards with just 500 episodes returns with the following values:

        | k     | mean      | std      |
        |-------|-----------|----------|
        | 500   | -6.258374 | 3.811000 |

As it can be seen, the mean is not within the margin of error of $\pm 0.05$. (This is an approximate statement, because we assumed that the true mean is close to the mean long-term discounted rewards with 30000 episodes, i.e. -6.376765) Thus, we need more episodes to get a 95% confidence that our mean is within $\pm 0.05$ of the true mean.

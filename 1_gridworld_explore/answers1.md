1. Following are different cases with different values of $\gamma$ and the optimal policy for each case.

    a. If we consider $\gamma = 0$, the optimal policy maximizes the immediate reward. In this case, the optimal policy is $\pi_\textt{{left}}$. This is because the reward for moving to the left in the first state is 1. Meanwhile, the other action (right) leads to an immediate reward of 0. Thus, the optimal policy is to move to the right at each step.

    b. If we consider $\gamma = 0.9$, the optimal policy is $\pi_\textt{{right}}$. This is because the reward for moving to the right in the first time step is 0, but the reward in the next timestep is 2, which when multiplied by 0.9 (the discount factor) leads to a total reward of 0 + 1.8. Meanwhile, the other action (left) leads to an immediate reward of 1 and a delayed reward of 0 (multiplied by the discount of 0.9). $\pi_\textt{{left}}$ gives a cumululative reward of 1. Thus, we choose the optimal policy as $\pi_\textt{{right}}$.

    c. If we consider $\gamma = 0.5$, either of the policies is optimal because in either case, we get the same reward. The optimal policy is $\pi_\textt{{left}}$ or $\pi_\textt{{right}}$.

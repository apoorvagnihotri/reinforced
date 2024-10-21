Using the following formula for the return of this system we can solve this exercise: $G_t = R_{t+1} + \gamma\cdot R_{t+2} $, where $R_{t+1}$ = 1 for the left action and 0 for the right action, and $R_{t+2}$ = 0 for the left action and 2 for the right action.
Following are different cases with different values of $\gamma$ and the optimal policy for each case.
    
a. If we consider $\gamma = 0$, the optimal policy maximizes the immediate reward as $G_t = R_{t+1}$. In this case, the optimal policy is $\pi_{left}$. This is because the reward for moving to the left in the first state is 1. Meanwhile, the other action (right) leads to an immediate reward of 0. Thus, the optimal policy is to move to the left at each step.
    
b. If we consider $\gamma = 0.9$, $G_t = R_{t+1} + 0.9\cdot R_{t+2}$. The optimal policy is $\pi_{right}$. This is because the reward for moving to the right in the first time step is 0, but the reward in the next timestep is 2, which when multiplied by 0.9 (the discount factor) leads to a total reward of 0 + 1.8. Meanwhile, the other action (left) leads to an immediate reward of 1 and a delayed reward of 0 (multiplied by the discount of 0.9). $\pi_{left}$ gives a cumulative reward of 1. Thus, we choose the optimal policy as $\pi_{right}$.
   
c. If we consider $\gamma = 0.5$, either of the policies is optimal because in either case, we get the same reward $G_t = 1 + 0.5\cdot0 = 0 + 0.5\cdot2 = 1$. The optimal policy is $\pi_{left}$ or $\pi_{right}$.




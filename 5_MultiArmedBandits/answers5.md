# Homework 5

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)


### Coding Question 3

Explore-Then-Commit (ETC) vs. ε-Greedy in Multi-Armed Bandits:

Explore-Then-Commit (ETC):
ETC splits the time horizon into two phases: exploration and exploitation. During exploration, it pulls each arm a fixed number of times to gather information about their expected rewards. Afterward, it commits to exploiting the arm with the highest estimated reward for the rest of the horizon. This method is simple but may suffer from suboptimal decisions if the exploration phase is too short.

ε-Greedy:
ε-greedy continuously balances exploration and exploitation throughout the time horizon. With probability ε, it explores by randomly selecting an arm, and with probability 1 − 𝜖, it exploits the arm with the highest estimated reward. It dynamically adapts and avoids rigid phase division, which makes it more flexible but potentially slower to converge to optimal decisions.

Comparison:
ETC is straightforward and computationally efficient but lacks adaptability since it cannot refine its estimates after committing. In contrast, ε-greedy allows ongoing exploration and adapts over time, making it more robust but slightly more complex in implementation.



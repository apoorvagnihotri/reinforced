# Homework 5

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)


### Coding Question 3


### Performance Comparison of Epsilon-Greedy, ETC, and UCB Strategies

## 1. Epsilon-Greedy
- **Performance**:  
  - Epsilon-greedy shows the **highest cumulative regret**, increasing steadily over time.
  - This is because epsilon-greedy balances exploration and exploitation **suboptimally**. It keeps exploring with a constant probability (epsilon = 0.1), even after the optimal arm is likely known, leading to unnecessary exploration in later stages.
  - The variance (shaded area) is also high, meaning its performance is less stable compared to other strategies.

---

## 2. Explore-Then-Commit (ETC)
- **Performance**:  
  - ETC outperforms epsilon-greedy with significantly lower cumulative regret.
  - ETC has a flat regret curve after its exploration phase (controlled by m = 10). Once it commits, it exploits the best arm without any further exploration, ensuring regret grows much slower compared to epsilon-greedy.
  - However, the regret during the exploration phase is higher compared to UCB because ETC performs exploration deterministically by sampling each arm m times regardless of how promising the arms appear.

---

## 3. UCB
- **Performance**:  
  - UCB (with different alpha values) shows the best performance overall, with the lowest cumulative regret. The regret grows very slowly over time.
  - UCB adapts exploration based on uncertainty, focusing more on underexplored arms early on and shifting to exploitation as confidence increases. This ensures efficient balancing of exploration and exploitation.
  - Among the UCB variants:
    - `alpha = 0.5`: Regret is higher compared to larger alpha values. This indicates insufficient exploration early on, leading to higher regret when suboptimal arms are committed early on.
    - `alpha = 4.0`: This variant achieves slightly better performance because the larger exploration bonus ensures more exploration early, reducing the risk of committing to suboptimal arms.
    - `alpha = 0.1`: Performs well but slightly worse than `alpha = 4.0` because it provides a smaller exploration bonus, leading to less aggressive exploration.

---

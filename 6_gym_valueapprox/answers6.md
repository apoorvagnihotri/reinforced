# Homework 6

#### Submission by: Apoorv Agnihotri (6604679), Gaurav Niranjan (6599177), Carla López Martínez (6637484)

## Q1. Function Approximation

#### a.
Tabular methods can be viewed as a special case of linear function approximation, where each state (or state-action pair) is represented by a one-hot encoded feature vector. Here's how:

Feature Vectors: For a state s in a state space with n states, the feature vector ϕ(s) is an n-dimensional vector, where:

ϕ(s)_i = {
    1 (if i corresponds to s.),
    0 (otherwise)
} 
 
Linear Function Approximation: The value function V(s) is represented as:

V(s)= w^⊤ . ϕ(s)

where w is a weight vector. Since ϕ(s) is one-hot encoded, V(s) directly corresponds to the weight w_i associated with state s.

Thus, tabular methods are a special case where the features are one-hot vectors, enabling exact representation of each state or state-action value.

#### b.


## Q2. Feature Designing

#### a.

One of the most basic way to encode the state is to directly take the pixel values and treat them as the feature vector.

## Q3. 





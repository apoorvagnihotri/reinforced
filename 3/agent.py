import util, random
from mdp import MarkovDecisionProcess


class Agent:

    def getAction(self, state):
        """
        For the given state, get the agent's chosen
        action.  The agent knows the legal actions
        """
        abstract

    def getValue(self, state):
        """
        Get the value of the state.
        """
        abstract

    def getQValue(self, state, action):
        """
        Get the q-value of the state action pair.
        """
        abstract

    def getPolicy(self, state):
        """
        Get the policy recommendation for the state.

        May or may not be the same as "getAction".
        """
        abstract

    def update(self, state, action, nextState, reward):
        """
        Update the internal state of a learning agent
        according to the (state, action, nextState)
        transistion and the given reward.
        """
        abstract


class RandomAgent(Agent):
    """
    Clueless random agent, used only for testing.
    """

    def __init__(self, actionFunction):
        self.actionFunction = actionFunction

    def getAction(self, state):
        return random.choice(self.actionFunction(state))

    def getValue(self, state):
        return 0.0

    def getQValue(self, state, action):
        return 0.0

    def getPolicy(self, state):
        return "random"

    def update(self, state, action, nextState, reward):
        pass


################################################################################
# Exercise 2

class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp: MarkovDecisionProcess = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()

        # Set value function of all states  = 0
        self.V = {s: 0 for s in states}

        for i in range(iterations):

            for state in states:

                if self.mdp.isTerminal(state):
                    continue

                v = self.V[state]

                # Bellman equation:
                self.V[state] = max(
                    sum(
                        prob
                        * (
                            self.mdp.getReward(state, action, next_state)
                            + discount * self.V[next_state]
                        )
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(
                            state, action
                        )
                    )
                    for action in self.mdp.getPossibleActions(state)
                )

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        return self.V[state]

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """

        q_pi_s_a = sum(
            prob
            * (
                self.mdp.getReward(state, action, next_state)
                + self.discount * self.V[next_state]
            )
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )

        return q_pi_s_a

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)

        best_action = actions[0]
        best_q_val = float("-inf")

        for action in actions:
            q_pi_s_a = self.getQValue(state, action)
            if q_pi_s_a >= best_q_val:
                best_q_val = q_pi_s_a
                best_action = action

        return best_action

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass


################################################################################
# Exercise 3

class PolicyIterationAgent(Agent):
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Initialize a PolicyIterationAgent with an MDP, discount factor, and number of iterations.
        """
        self.mdp: MarkovDecisionProcess = mdp
        self.discount = discount
        self.iterations = iterations
        
        # Initialize value function and policy for all states
        states = self.mdp.getStates()
        self.V = {s: 0 for s in states}
        self.policy = {s: self.mdp.getPossibleActions(s)[0] if not self.mdp.isTerminal(s) and self.mdp.getPossibleActions(s) else None for s in states}
        
        # Run policy iteration
        for i in range(iterations):
            # Policy evaluation
            old_V = self.V.copy()
            stable = True
            
            # Keep evaluating until convergence
            while True:
                delta = 0
                for state in states:
                    if self.mdp.isTerminal(state):
                        continue
                        
                    v = self.V[state]
                    # Evaluate current policy
                    action = self.policy[state]
                    self.V[state] = sum(
                        prob * (self.mdp.getReward(state, action, next_state) + 
                               discount * self.V[next_state])
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
                    )
                    delta = max(delta, abs(v - self.V[state]))
                
                if delta < 1e-6:  # Small threshold for convergence
                    break
            
            # Policy improvement
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                    
                old_action = self.policy[state]
                actions = self.mdp.getPossibleActions(state)
                
                # Find best action according to current values
                best_action = max(
                    actions,
                    key=lambda a: sum(
                        prob * (self.mdp.getReward(state, a, next_state) + 
                               discount * self.V[next_state])
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, a)
                    )
                )
                
                self.policy[state] = best_action
                if old_action != best_action:
                    stable = False
            
            # If policy is stable, we've converged
            if stable:
                break

    def getValue(self, state):
        """
        Return the value of the state after policy iteration.
        """
        return self.V[state]

    def getQValue(self, state, action):
        """
        Return the Q-value of the state-action pair.
        """
        return sum(
            prob * (self.mdp.getReward(state, action, next_state) + 
                   self.discount * self.V[next_state])
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )

    def getPolicy(self, state):
        """
        Return the policy's action for the state.
        """
        return self.policy[state]

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents.
        """
        pass



################################################################################
# Below can be ignored for Exercise 2


class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.2):
        """
        A Q-Learning agent gets nothing about the mdp on
        construction other than a function mapping states to actions.
        The other parameters govern its exploration
        strategy and learning rate.
        """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        raise "Your code here."

    # THESE NEXT METHODS ARE NEEDED TO WIRE YOUR AGENT UP TO THE CRAWLER GUI

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    # GENERAL RL AGENT METHODS

    def getValue(self, state):
        """
        Look up the current value of the state.
        """

        raise ValueError("Your code here.")

    def getQValue(self, state, action):
        """
        Look up the current q-value of the state action pair.
        """

        raise ValueError("Your code here.")

    def getPolicy(self, state):
        """
        Look up the current recommendation for the state.
        """

        raise ValueError("Your code here.")

    def getAction(self, state):
        """
        Choose an action: this will require that your agent balance
        exploration and exploitation as appropriate.
        """

        raise ValueError("Your code here.")

    def update(self, state, action, nextState, reward):
        """
        Update parameters in response to the observed transition.
        """

        raise ValueError("Your code here.")

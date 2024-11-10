import util, random
from collections import defaultdict


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

  def reset(self):
    """
    called to reset the agent at the beginning of an episode
    """
    pass


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
    return 'random'

  def update(self, state, action, nextState, reward):
    pass

################################################################################
# Exercise 2

class ValueIterationAgent(Agent):

  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations

    states = self.mdp.getStates()
    number_states = len(states)

    self.V = { s: 0 for s in states }


    for i in range(iterations):
      newV = {}
      for s in states:
        actions = self.mdp.getPossibleActions(s)
        if len(actions) < 1:
          newV[s] = 0.0
        else:
          newV[s] = np.max([ self.mdp.getReward(s, a, None) + \
                             self.discount * np.sum([prob * self.V[sp]
                                                     for sp, prob in self.mdp.getTransitionStatesAndProbs(s, a) ])
                             for a in actions ])
      self.V = newV


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
    # get all successor states and probabilities and evaluate value of these states
    return self.mdp.getReward(state, action, None) + \
      self.discount *  np.sum([self.V[sp] * prob for sp, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

  def getPolicy(self, state):
    """
    Look up the policy's recommendation for the state
    (after the indicated number of value iteration passes).
    """
    # do greedy on Q
    actions = self.mdp.getPossibleActions(state)
    if len(actions) < 1:
      return None
    else:
      qValues = [self.getQValue(state, a) for a in actions]
      action_index = np.argmax(qValues)
      return actions[action_index]

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
# Exercise 4

class QLearningAgent(Agent):

  def __init__(self, actionFunction, discount = 0.9, learningRate = 0.1, epsilon = 0.2):
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

    self.qValues = defaultdict(float)   #holds Q-values; keys are (state, action) pairs






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

    actions = self.actionFunction(state)

    if not actions:  #If no actions available
      return 0.0   
    
    return max(self.getQValue(state,action) for action in actions)




  def getQValue(self, state, action):
    """
    Look up the current q-value of the state action pair.
    """


    return self.qValues[(state,action)]




  def getPolicy(self, state):
    """
    Look up the current recommendation for the state.
    """


    actions = self.actionFunction(state)

    if not actions:
      return 0.0
    
    #Action with the highest Q-Value
    best_action = max(actions, key=lambda action: self.getQValue(state,action))

    return best_action




  def getAction(self, state):
    """
    Choose an action: this will require that your agent balance
    exploration and exploitation as appropriate.
    """

    actions = self.actionFunction(state)

    if not actions:
      return None
    
    if random.random() < self.epsilon:
      return random.choice(actions)
    else:
      return self.getPolicy(state)




  def update(self, state, action, nextState, reward):
    """
    Update parameters in response to the observed transition.
    """

    sample = reward + self.discount * self.getValue(nextState)  #here getValue returns the maxQ(s',a) over all actions a 

    old_q_val = self.getQValue(state, action)
    new_q_val = old_q_val + self.learningRate*(sample-old_q_val)

    self.qValues[(state,action)] = new_q_val

  def reset(self):
    """
    called to reset the agent at the beginning of an episode
    """
    pass

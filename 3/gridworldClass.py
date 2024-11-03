import mdp
import util

class Gridworld(mdp.MarkovDecisionProcess):

  def __init__(self, grid):
    # layout
    self.grid = grid
    self.rows = len(grid)
    self.cols = len(grid[0])
    self.terminalState = (-1, -1)

    # parameters
    self.livingReward = 0.0
    self.noise = 0.2

  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.

    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward

  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise


  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.

    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    if state == self.terminalState:
      return ()
    row, col = state
    if type(self.grid[row][col]) == int:
      return ('exit',)
    return ('north','west','south','east')


  def getStates(self):
    """
    Return list of all states.
    """
    states = [self.terminalState]  # The true terminal state.
    for row in range(self.rows):
      for col in range(self.cols):
        if self.grid[row][col] != '#':
          states.append((row, col))
    return states

  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.

    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if state == self.terminalState:
      return 0.0
    row, col = state
    cell = self.grid[row][col]
    if type(cell) == int or type(cell) == float:
      return cell
    return self.livingReward

  def getStartState(self):
    for row in range(self.rows):
      for col in range(self.cols):
        if self.grid[row][col] == 'S':
          return (row, col)
    raise 'Grid has no start state'

  def isTerminal(self, state):
    """
    Only the (-1, -1) state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return state == self.terminalState


  def getTransitionStatesAndProbs(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.
    """

    if action not in self.getPossibleActions(state):
      raise "Illegal action!"

    if state == self.terminalState:
      return []

    row, col = state

    if type(self.grid[row][col]) == int or type(self.grid[row][col]) == float:
      return [(self.terminalState, 1.0)]

    successors = []

    action2state = {
      'north' : (self.__isAllowed(row-1,col) and (row-1,col)) or state,
      'west'  : (self.__isAllowed(row,col-1) and (row,col-1)) or state,
      'south' : (self.__isAllowed(row+1,col) and (row+1,col)) or state,
      'east'  : (self.__isAllowed(row,col+1) and (row,col+1)) or state
    }

    num_actions = len(action2state)
    for a,s in action2state.items():
      if a != action:
        successors.append((s,self.noise/(num_actions-1)))

    successors.append((action2state[action],1.0-self.noise))

    successors = self.__aggregate(successors)

    return successors

  def __aggregate(self, statesAndProbs):
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter.incrementCount(state, prob)
    newStatesAndProbs = []
    # print counter
    for state, prob in list(counter.items()):
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs

  def __isAllowed(self, row, col):
    if row < 0 or row >= self.rows: return False
    if col < 0 or col >= self.cols: return False
    return self.grid[row][col] != '#'

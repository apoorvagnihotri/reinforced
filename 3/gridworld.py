import random
import sys
import mdp
import environment
import util
import optparse

from gridworldClass import Gridworld

class GridworldEnvironment(environment.Environment):

  def __init__(self, gridWorld):
    self.gridWorld = gridWorld
    self.reset()

  def getCurrentState(self):
    return self.state

  def getPossibleActions(self, state):
    return self.gridWorld.getPossibleActions(state)

  def doAction(self, action):
    successors = self.gridWorld.getTransitionStatesAndProbs(self.state, action)
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.'
      if rand < sum:
        reward = self.gridWorld.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'

  def reset(self):
    self.state = self.gridWorld.getStartState()


def getCliffGrid():
  grid = [[' ',' ',' ',' ',' '],
          ['S',' ',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(grid)

def getCliffGrid2():
  grid = [[' ',' ',' ',' ',' '],
          [8,'S',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(grid)

def getDiscountGrid():
  grid = [[' ',' ',' ',' ',' '],
          [' ','#',' ',' ',' '],
          [' ','#', 1,'#', 10],
          ['S',' ',' ',' ',' '],
          [-10,-10, -10, -10, -10]]
  return Gridworld(grid)

def getBridgeGrid():
  grid = [[ '#',-100, -100, -100, -100, -100, '#'],
          [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
          [ '#',-100, -100, -100, -100, -100, '#']]
  return Gridworld(grid)

def getBookGrid():
  grid = [[' ',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid)

def getMazeGrid():
  grid = [[' ',' ',' ',+1],
          ['#','#',' ','#'],
          [' ','#',' ',' '],
          [' ','#','#',' '],
          ['S',' ',' ',' ']]
  return Gridworld(grid)

def getCircularGrid():
  grid = [['#','#','#','#',+10],
          [' ',' ',' ','#',' '],
          [' ','#',' ','#',' '],
          [' ','#','S','#',' '],
          [' ','#','#','#',' '],
          [' ',' ',' ',' ',' ']]
  return Gridworld(grid)



def getUserAction(state, actionFunction):
  """
  Get an action from the user (rather than the agent).

  Used for debugging and lecture demos.
  """
  import graphicsUtils
  action = None
  while True:
    keys = graphicsUtils.wait_for_keys()
    if 'Up' in keys: action = 'north'
    if 'Down' in keys: action = 'south'
    if 'Left' in keys: action = 'west'
    if 'Right' in keys: action = 'east'
    if 'q' in keys: sys.exit(0)
    if action == None: continue
    break
  actions = actionFunction(state)
  if action not in actions:
    action = actions[0]
  return action

def printString(x):
  print(x)

def runEpisode(agent, environment, discount, decision, display, message, pause, episode):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  message("BEGINNING EPISODE: "+str(episode)+"\n")
  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
    display(state)
    pause()

    # END IF IN A TERMINAL STATE
    actions = environment.getPossibleActions(state)
    if len(actions) == 0:
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
      return returns

    # GET ACTION (USUALLY FROM AGENT)
    action = decision(state)
    if action == None:
      raise 'Error: Agent returned None action'

    # EXECUTE ACTION
    nextState, reward = environment.doAction(action)
    message("Started in state: "+str(state)+
            "\nTook action: "+str(action)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")

    # UPDATE LEARNER
    agent.update(state, action, nextState, reward)
    returns += reward * totalDiscount
    totalDiscount *= discount



def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=0,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\', \'policy\' and \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent (for lecture)')

    opts, args = optParser.parse_args()

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
      opts.pause = False
      opts.manual = False

    if opts.manual:
      opts.pause = False

    return opts


if __name__ == '__main__':

  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  import gridworld
  mdpFunction = getattr(gridworld, "get"+opts.grid)
  mdp = mdpFunction()
  mdp.setLivingReward(opts.livingReward)
  mdp.setNoise(opts.noise)
  env = gridworld.GridworldEnvironment(mdp)


  ###########################
  # GET THE DISPLAY ADAPTER
  ###########################

  import textGridworldDisplay
  display = textGridworldDisplay.TextGridworldDisplay(mdp)
  if not opts.textDisplay:
    import graphicsGridworldDisplay
    display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)
  display.start()

  ###########################
  # GET THE AGENT
  ###########################

  import agent
  a = None
  if opts.agent == 'value':
    a = agent.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'q':
    a = agent.QLearningAgent(env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon)
  elif opts.agent == 'random':
    # No reason to use the random agent without episodes
    if opts.episodes == 0:
      opts.episodes = 1
    a = agent.RandomAgent(mdp.getPossibleActions)
  elif opts.agent == 'policy':
    a = agent.PolicyIterationAgent(mdp, opts.discount, opts.iters)
  else:
    raise 'Unknown agent type: '+opts.agent

  ###########################
  # RUN EPISODES
  ###########################

  # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
  if opts.agent == 'value':
    display.displayValues(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()
    display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()
  
  if opts.agent == 'policy':
    display.displayValues(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()

  # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  displayCallback = lambda x: None
  if not opts.quiet:
    if opts.agent == 'random': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
    if opts.agent == 'value': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
    if opts.agent == 'q': displayCallback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")
    if opts.agent == 'policy': displayCallback = lambda state: display.displayValues(a, state, "CURRENT POLICY W/ VALUES")

  messageCallback = lambda x: printString(x)
  if opts.quiet:
    messageCallback = lambda x: None

  # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
  pauseCallback = lambda : None
  if opts.pause:
    pauseCallback = lambda : display.pause()

  # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)
  decisionCallback = a.getAction
  if opts.manual:
    decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)

  # RUN EPISODES
  if opts.episodes > 0:
    print()
    print("RUNNING", opts.episodes, "EPISODES")
    print()
  returns = 0
  for episode in range(1, opts.episodes+1):
    returns += runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print()
    print("AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes))
    print()
    print()

  # DISPLAY POST-LEARNING VALUES / Q-VALUES
  if opts.agent == 'q' and not opts.manual:
    display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
    display.pause()
    display.displayValues(a, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
    display.pause()

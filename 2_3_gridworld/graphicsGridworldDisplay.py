import util
from graphicsUtils import *
from functools import reduce


class GraphicsGridworldDisplay:
  
  def __init__(self, gridworld, size=120, speed=1.0):
    self.gridworld = gridworld
    self.size = size
    self.speed = speed
    
  def start(self):
    setup(self.gridworld, size=self.size)
  
  def pause(self):
    wait_for_keys()
  
  def displayValues(self, agent, currentState = None, message = 'Agent Values'):
    values = util.Counter()
    policy = {}
    states = self.gridworld.getStates()
    for state in states:
      values.setCount(state, agent.getValue(state))
      policy[state] = agent.getPolicy(state)
    drawValues(self.gridworld, values, policy, currentState, message)
    sleep(0.05 / self.speed)

  def displayQValues(self, agent, currentState = None, message = 'Agent Q-Values'):
    qValues = util.Counter()
    states = self.gridworld.getStates()
    for state in states:
      for action in self.gridworld.getPossibleActions(state):
        qValues.setCount((state, action), agent.getQValue(state, action))
    drawQValues(self.gridworld, qValues, currentState, message)
    sleep(0.05 / self.speed)



BACKGROUND_COLOR = formatColor(0,0,0)    
EDGE_COLOR = formatColor(1,1,1)
OBSTACLE_COLOR = formatColor(0.5,0.5,0.5)
TEXT_COLOR = formatColor(1,1,1)
LOCATION_COLOR = formatColor(0,0,1)

WINDOW_SIZE = -1
GRID_SIZE = -1
MARGIN = -1

def setup(gridworld, title = "Gridworld Display", size = 120):
  global GRID_SIZE, MARGIN, SCREEN_WIDTH, SCREEN_HEIGHT
  grid = gridworld.grid
  width = len(grid[0]) - 1
  height = len(grid) - 1 + 0.5
  WINDOW_SIZE = size
  GRID_SIZE = size
  MARGIN = GRID_SIZE * 0.75
  screen_width = width * GRID_SIZE + MARGIN * 2
  screen_height = height * GRID_SIZE + MARGIN * 2

  begin_graphics(screen_width,    
                 screen_height,
                 BACKGROUND_COLOR, title=title)

def drawValues(gridworld, values, policy, currentState = None, message = 'State Values'):
  grid = gridworld.grid
  blank()
  valueList = [values.getCount(state) for state in gridworld.getStates()] + [0.0]
  minValue = min(valueList)
  maxValue = max(valueList)
  for row in range(gridworld.rows):
    for col in range(gridworld.cols):
      state = (row, col)
      gridType = grid[row][col]
      isExit = (str(gridType) != gridType)
      isCurrent = (currentState == state)
      if gridType == '#':
        drawSquare(row, col, 0, 0, 0, None, None, True, False, isCurrent)
      else:
        value = values.getCount(state)
        action = None
        if policy != None and state in policy:
          action = policy[state]
          actions = gridworld.getPossibleActions(state)
        if action not in actions and 'exit' in actions:
          action = 'exit'
        valString = '%.2f' % value
        drawSquare(row, col, value, minValue, maxValue, valString, action, False, isExit, isCurrent)
  text( to_screen((len(grid)-0.2, (len(grid[0])-1.0) / 2.0)), TEXT_COLOR, message, "Courier", 12, "bold", "c")

def drawQValues(gridworld, qValues, currentState = None, message = 'State-Action Q-Values'):
  grid = gridworld.grid
  blank()
  stateCrossActions = [[(state, action) for action in gridworld.getPossibleActions(state)] for state in gridworld.getStates()]
  qStates = reduce(lambda x,y: x+y, stateCrossActions, [])
  qValueList = [qValues.getCount((state, action)) for state, action in qStates] + [0.0]
  minValue = min(qValueList)
  maxValue = max(qValueList)
  for row in range(gridworld.rows):
    for col in range(gridworld.cols):
      state = (row, col)
      gridType = grid[row][col]
      isExit = (str(gridType) != gridType)
      isCurrent = (currentState == state)
      actions = gridworld.getPossibleActions(state)
      if actions == None or len(actions) == 0:
        actions = [None]
      bestQ = max([qValues.getCount((state, action)) for action in actions])
      bestActions = [action for action in actions if qValues.getCount((state, action)) == bestQ]

      q = util.Counter()
      valStrings = {}
      for action in actions:
        v = qValues.getCount((state, action))
        q.incrementCount(action, v)
        valStrings[action] = '%.2f' % v
      if gridType == '#':
        drawSquare(row, col, 0, 0, 0, None, None, True, False, isCurrent)
      elif isExit:
        action = 'exit'
        value = q.getCount(action)
        valString = '%.2f' % value
        drawSquare(row, col, value, minValue, maxValue, valString, action, False, isExit, isCurrent)
      else:
        drawSquareQ(row, col, q, minValue, maxValue, valStrings, actions, isCurrent)
  text( to_screen((len(grid)-0.2, (len(grid[0])-1.0) / 2.0)), TEXT_COLOR, message, "Courier", 12, "bold", "c")


def blank():
  clear_screen()
      
def drawSquare(row, col, val, min, max, valStr, action, isObstacle, isTerminal, isCurrent):

  square_color = getColor(val, min, max)
  
  if isObstacle:
    square_color = OBSTACLE_COLOR
    
  (screen_x, screen_y) = to_screen((row, col))
  square( (screen_x, screen_y), 
                 0.5* GRID_SIZE, 
                 color = square_color,
                 filled = 1,
                 width = 1)
  
  square( (screen_x, screen_y), 
                 0.5* GRID_SIZE, 
                 color = EDGE_COLOR,
                 filled = 0,
                 width = 3)
  
  if isTerminal and not isObstacle:
    square( (screen_x, screen_y), 
                 0.4* GRID_SIZE, 
                 color = EDGE_COLOR,
                 filled = 0,
                 width = 2)
    
  if action == 'north':
    polygon( [(screen_x, screen_y - 0.45*GRID_SIZE), (screen_x+0.05*GRID_SIZE, screen_y-0.40*GRID_SIZE), (screen_x-0.05*GRID_SIZE, screen_y-0.40*GRID_SIZE)], EDGE_COLOR, filled = 1, smooth = 0)
  if action == 'south':
    polygon( [(screen_x, screen_y + 0.45*GRID_SIZE), (screen_x+0.05*GRID_SIZE, screen_y+0.40*GRID_SIZE), (screen_x-0.05*GRID_SIZE, screen_y+0.40*GRID_SIZE)], EDGE_COLOR, filled = 1, smooth = 0)
  if action == 'west':
    polygon( [(screen_x-0.45*GRID_SIZE, screen_y), (screen_x-0.4*GRID_SIZE, screen_y+0.05*GRID_SIZE), (screen_x-0.4*GRID_SIZE, screen_y-0.05*GRID_SIZE)], EDGE_COLOR, filled = 1, smooth = 0)
  if action == 'east':
    polygon( [(screen_x+0.45*GRID_SIZE, screen_y), (screen_x+0.4*GRID_SIZE, screen_y+0.05*GRID_SIZE), (screen_x+0.4*GRID_SIZE, screen_y-0.05*GRID_SIZE)], EDGE_COLOR, filled = 1, smooth = 0)
    
  
  text_color = TEXT_COLOR

  if not isObstacle and isCurrent:
    circle( (screen_x, screen_y), 0.1*GRID_SIZE, LOCATION_COLOR, filled = 0 )

  if not isObstacle:
    text( (screen_x, screen_y), text_color, valStr, "Courier", 12, "bold", "c")


def drawSquareQ(row, col, qVals, min, max, valStrs, bestActions, isCurrent):

  (screen_x, screen_y) = to_screen((row, col))
  
  center = (screen_x, screen_y)
  nw = (screen_x-0.5*GRID_SIZE, screen_y-0.5*GRID_SIZE)
  ne = (screen_x+0.5*GRID_SIZE, screen_y-0.5*GRID_SIZE)
  se = (screen_x+0.5*GRID_SIZE, screen_y+0.5*GRID_SIZE)
  sw = (screen_x-0.5*GRID_SIZE, screen_y+0.5*GRID_SIZE)
  n = (screen_x, screen_y-0.5*GRID_SIZE+5)
  s = (screen_x, screen_y+0.5*GRID_SIZE-5)
  w = (screen_x-0.5*GRID_SIZE+5, screen_y)
  e = (screen_x+0.5*GRID_SIZE-5, screen_y)
  
  actions = list(qVals.keys())
  for action in actions:
    
    wedge_color = getColor(qVals.getCount(action), min, max)

    if action == 'north':
      polygon( (center, nw, ne), wedge_color, filled = 1, smooth = 0)
      #text(n, text_color, valStr, "Courier", 8, "bold", "n")
    if action == 'south':
      polygon( (center, sw, se), wedge_color, filled = 1, smooth = 0)
      #text(s, text_color, valStr, "Courier", 8, "bold", "s")
    if action == 'east':
      polygon( (center, ne, se), wedge_color, filled = 1, smooth = 0)
      #text(e, text_color, valStr, "Courier", 8, "bold", "e")
    if action == 'west':
      polygon( (center, nw, sw), wedge_color, filled = 1, smooth = 0)
      #text(w, text_color, valStr, "Courier", 8, "bold", "w")
      
  square( (screen_x, screen_y), 
                 0.5* GRID_SIZE, 
                 color = EDGE_COLOR,
                 filled = 0,
                 width = 3)
  line(ne, sw, color = EDGE_COLOR)
  line(nw, se, color = EDGE_COLOR)

  if isCurrent:
    circle( (screen_x, screen_y), 0.1*GRID_SIZE, LOCATION_COLOR, filled = 0 )

  for action in actions:
    text_color = TEXT_COLOR
    valStr = ""
    if action in valStrs:
      valStr = valStrs[action]
    h = 10
    if action == 'north':
      #polygon( (center, nw, ne), wedge_color, filled = 1, smooth = 0)
      text(n, text_color, valStr, "Courier", h, "bold", "n")
    if action == 'south':
      #polygon( (center, sw, se), wedge_color, filled = 1, smooth = 0)
      text(s, text_color, valStr, "Courier", h, "bold", "s")
    if action == 'east':
      #polygon( (center, ne, se), wedge_color, filled = 1, smooth = 0)
      text(e, text_color, valStr, "Courier", h, "bold", "e")
    if action == 'west':
      #polygon( (center, nw, sw), wedge_color, filled = 1, smooth = 0)
      text(w, text_color, valStr, "Courier", h, "bold", "w")


def getColor(val, min, max):
  r, g = 0.0, 0.0
  if val < 0 and min < 0:
    r = val * 0.65 / min
  if val > 0 and max > 0:
    g = val * 0.65 / max
  return formatColor(r,g,0.0)


def square(pos, size, color, filled, width):
  x, y = pos
  dx, dy = size, size
  return polygon([(x - dx, y - dy), (x - dx, y + dy), (x + dx, y + dy), (x + dx, y - dy)], color, filled, smooth = 0, width=width)
  
  
def to_screen(point):
  ( row, col ) = point
  x = col*GRID_SIZE + MARGIN  
  y = row*GRID_SIZE + MARGIN  
  return ( x, y )

def to_grid(point):
  (x, y) = point
  row = int ((y - MARGIN + GRID_SIZE * 0.5) / GRID_SIZE)
  col = int ((x - MARGIN + GRID_SIZE * 0.5) / GRID_SIZE)
  print(point, "-->", (row, col))
  return (row, col)



# TEST OF DISPLAY CODE
                                
if __name__ == '__main__':
  import gridworld
  
  grid = gridworld.getCliffGrid3()
  print(grid.getStates())
  
  setup(grid)
  
  policy = dict([(state,'east') for state in grid.getStates()])
  values = util.Counter(dict([(state,1000.23) for state in grid.getStates()]))
  drawValues(grid, values, policy, currentState = (0,0))
  wait_for_keys()

  stateCrossActions = [[(state, action) for action in grid.getPossibleActions(state)] for state in grid.getStates()]
  qStates = reduce(lambda x,y: x+y, stateCrossActions, [])
  qValues = util.Counter(dict([((state, action), 10.5) for state, action in qStates]))
  drawQValues(grid, qValues, currentState = (0,0))
  wait_for_keys()
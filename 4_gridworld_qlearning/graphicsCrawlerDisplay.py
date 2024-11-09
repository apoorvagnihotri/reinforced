import tkinter
import agent
import time
import threading
import sys
import crawler
#import pendulum
import math
from math import pi as PI

robotType = 'crawler'

class Application:

    def incrementEpsilon(self, inc):
        self.epsilon += inc
        self.epsilon = min(1.0, self.epsilon)
        self.epsilon = max(0.0,self.epsilon)
        self.learner.setEpsilon(self.epsilon)
        self.epsilon_label['text'] = 'Epsilon: %.2f' % (self.epsilon)

    def incrementGamma(self, inc):
        if inc > 0:
            self.gamma *= 1.01
        else:
            self.gamma *= 1.0/1.01

        self.gamma = min(1.0, self.gamma)
        self.gamma = max(0.0,self.gamma)
        self.learner.setDiscount(self.gamma)
        self.gamma_label['text'] = 'Discount: %.4f' % (self.gamma)

    def incrementAlpha(self, inc):
        self.alpha += inc
        self.alpha = min(1.0, self.alpha)
        self.alpha = max(0.0, self.alpha)
        self.learner.setLearningRate(self.alpha)
        self.alpha_label['text'] = 'Learning Rate: %.2f' % (self.alpha)

    def __initGUI(self, win):
        ## Window ##
        self.win = win

        ## Initialize Frame ##
        win.grid()

        ## Epsilon Button + Label ##
        self.setupEpsilonButtonAndLabel(win)

        ## Gamma Button + Label ##
        self.setUpGammaButtonAndLabel(win)

        ## Alpha Button + Label ##
        self.setupAlphaButtonAndLabel(win)

        ## RewardLabel ##
        self.setupRewardLabel(win)

        ## Exit Button ##
        self.exit_button = tkinter.Button(win,text='Quit', command=self.exit)
        self.exit_button.grid(row=0, column=11)

        ## Simulation Buttons ##
        self.setupSimulationButtons(win)
        self.tickTime = 0.1

         ## Canvas ##
        self.canvas = tkinter.Canvas(root, height=200, width=1280)
        self.canvas.grid(row=2,columnspan=12)

    def setupAlphaButtonAndLabel(self, win):
        self.alpha_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementAlpha(-0.1)))
        self.alpha_minus.grid(row=0, column=7, padx=5)

        self.alpha_label = tkinter.Label(win, text='Learning Rate: %.2f' % (self.alpha))
        self.alpha_label.grid(row=0, column=8)

        self.alpha_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementAlpha(0.1)))
        self.alpha_plus.grid(row=0, column=9, padx=5)

    def setUpGammaButtonAndLabel(self, win):
        self.gamma_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementGamma(-0.1)))
        self.gamma_minus.grid(row=0, column=4, padx=5)

        self.gamma_label = tkinter.Label(win, text='Discount: %.4f' % (self.gamma))
        self.gamma_label.grid(row=0, column=5)

        self.gamma_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementGamma(0.1)))
        self.gamma_plus.grid(row=0, column=6, padx=5)

    def setupEpsilonButtonAndLabel(self, win):
        self.epsilon_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementEpsilon(-0.1)))
        self.epsilon_minus.grid(row=0, column=1)

        self.epsilon_label = tkinter.Label(win, text='Epsilon: %.2f' % (self.epsilon))
        self.epsilon_label.grid(row=0, column=2)

        self.epsilon_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementEpsilon(0.1)))
        self.epsilon_plus.grid(row=0, column=3)


    def setupRewardLabel(self, win):
        self.reward_label = tkinter.Label(win, text='I: 0 R-avg: %.3f' % (self.average_reward))
        self.reward_label.grid(row=0, column=10, padx=5)

    def setupSimulationButtons(self, win):
        self.skip5kStepsButton =  tkinter.Button(win,
        text='Skip 5000 Steps', command=self.skip5kSteps)
        self.skip5kStepsButton.grid(row=0, column=0)

        if robotType == 'pendulum':
            self.animatePolicyButton = tkinter.Button(win,
            text='Animate Policy', command=self.animatePolicy)
            self.animatePolicyButton.grid(row=1,column=3,columnspan=3)

    def skip5kSteps(self):
        self.stepsToSkip = 5000

    def __init__(self, win):

        self.epsilon = 0.1
        self.gamma = 0.8
        self.alpha = 0.9

        self.average_reward = 0

        ## Init Gui
        self.__initGUI(win)

        # Init environment
        if robotType == 'crawler':
            self.robot = crawler.CrawlingRobot(self.canvas)
            self.robotEnvironment = crawler.CrawlingRobotEnvironment(self.robot)
        elif robotType == 'pendulum':
            self.robot = pendulum.PendulumRobot(self.canvas)
            self.robotEnvironment = \
                pendulum.PendulumRobotEnvironment(self.robot)
        else:
            raise "Unknown RobotType"


        # Init Agent

        self.learner = agent.QLearningAgent(self.robotEnvironment.getPossibleActions)

        self.learner.setEpsilon(self.epsilon)
        self.learner.setLearningRate(self.alpha)
        self.learner.setDiscount(self.gamma)

        # Start GUI
        self.running = True
        self.stopped = False
        self.stepsToSkip = 0
        self.thread = threading.Thread(target=self.run)
        self.thread.start()


    def exit(self):
      self.running = False
      for i in range(5):
        if not self.stopped:
          time.sleep(0.1)
      self.win.destroy()
      sys.exit(0)

    def step(self):

        self.stepCount += 1

        state = self.robotEnvironment.getCurrentState()
        actions = self.robotEnvironment.getPossibleActions(state)
        if len(actions) == 0.0:
            self.robotEnvironment.reset()
            state = self.robotEnvironment.getCurrentState()
            actions = self.robotEnvironment.getPossibleActions(state)
            print('Reset!')
        action = self.learner.getAction(state)
        if action == None:
            raise ValueError('None action returned: Code Not Complete')
        nextState, reward = self.robotEnvironment.doAction(action)
        self.average_reward = 0.99*self.average_reward + 0.01*reward        # added
        self.learner.update(state, action, nextState, reward)

    def animatePolicy(self):
        if robotType != 'pendulum':
            raise ValueError('Only pendulum can animatePolicy')


        totWidth = self.canvas.winfo_reqwidth()
        totHeight = self.canvas.winfo_reqheight()

        length = 0.48 * min(totWidth, totHeight)
        x,y = totWidth-length-30, length+10



        angleMin, angleMax = self.robot.getMinAndMaxAngle()
        velMin, velMax = self.robot.getMinAndMaxAngleVelocity()

        if not 'animatePolicyBox' in dir(self):
            self.canvas.create_line(x,y,x+length,y)
            self.canvas.create_line(x+length,y,x+length,y-length)
            self.canvas.create_line(x+length,y-length,x,y-length)
            self.canvas.create_line(x,y-length,x,y)
            self.animatePolicyBox = 1
            self.canvas.create_text(x+length/2,y+10,text='angle')
            self.canvas.create_text(x-30,y-length/2,text='velocity')
            self.canvas.create_text(x-60,y-length/4,text='Blue = kickLeft')
            self.canvas.create_text(x-60,y-length/4+20,text='Red = kickRight')
            self.canvas.create_text(x-60,y-length/4+40,text='White = doNothing')



        angleDelta = (angleMax-angleMin) / 100
        velDelta = (velMax-velMin) / 100
        for i in range(100):
            angle = angleMin + i * angleDelta

            for j in range(100):
                vel = velMin + j * velDelta
                state = self.robotEnvironment.getState(angle,vel)
                max, argMax = None, None
                if not self.learner.seenState(state):
                    argMax = 'unseen'
                else:
                     for action in ('kickLeft','kickRight','doNothing'):
                         qVal = self.learner.getQValue(state, action)
                         if max == None or qVal > max:
                             max, argMax = qVal, action
                if argMax != 'unseen':
                    if argMax == 'kickLeft':
                        color = 'blue'
                    elif argMax == 'kickRight':
                        color = 'red'
                    elif argMax == 'doNothing':
                        color = 'white'
                    dx = length / 100.0
                    dy = length / 100.0
                    x0, y0 = x+i*dx, y-j*dy
                    self.canvas.create_rectangle(x0,y0,x0+dx,y0+dy,fill=color)




    def run(self):
        self.stepCount = 0
        while True:
          time.sleep(self.tickTime)
          if not self.running:
            self.stopped = True
            return
          for i in range(self.stepsToSkip):
              self.step()
          self.stepsToSkip = 0
          self.step()
          self.robot.draw()
          self.reward_label['text'] = 'I: %i R-avg: %.3f' % (self.stepCount, self.average_reward)

    def start(self):
        self.win.mainloop()



def run():
  global root
  root = tkinter.Tk()
  root.title( 'Crawler GUI' )
  root.resizable( 0, 0 )

  app = Application(root)
  root.protocol( 'WM_DELETE_WINDOW', app.exit)
  app.start()

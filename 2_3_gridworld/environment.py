#!/usr/bin/python

class Environment:
        
  def getCurrentState(self):
    """
    Returns the current state of enviornment
    """
    abstract
    
  def getPossibleActions(self, state):
    """
    Returns possible actions the agent 
    can take in the given state. Can
    return the empty list if we are in 
    a terminal state.
    """
    abstract
                
  def doAction(self, action):
    """
    Performs the given action in the current
    environment state and updates the enviornment.
    
    Returns a (reward, nextState) pair
    """
    abstract
        
  def reset(self):
    """
    Resets the current state to the start state
    """
    abstract
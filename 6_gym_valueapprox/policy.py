import numpy as np
import math

class PDPolicy:
    def __init__(self, env, swingup=True, eps_expl=0.2):
        self.K = np.asarray([[10,  2.0]])
        self.eps_expl = eps_expl
        self.env = env
        self.swingup = swingup

    def get_action(self, obs):

        x, y, thdot = obs
        th = math.atan2(y,x)

        if th < -np.pi/2 and self.swingup:
            u = [np.sign(thdot) - .01*thdot]
        elif th > np.pi/2 and self.swingup:
            u = [np.sign(thdot)  - .01*thdot]
        else:
            u = np.dot(-self.K, [th, thdot])
        if np.random.rand() < self.eps_expl:
            u = self.env.action_space.sample()

        return np.clip(u,self.env.action_space.low,self.env.action_space.high)

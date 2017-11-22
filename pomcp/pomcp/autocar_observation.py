import sys
import numpy as np

class AutocarObservation():
    def __init__(self, state):
        self.obs = [[[0 for x in range(len(state[0][0]))] for y in range(len(state[0]))] for z in range(len(state))]
        for i in range(len(state)):
            for j in range(len(state[0])):
                for k in range(len(state[0][0])):
                    if state[i][j][k] == 1:
                        if j > 1:
                            self.obs[i][j-1][k] = 0.1
                        self.obs[i][j][k] = 0.8
                        if j < 99:
                            self.obs[i][j+1][k] = 0.1


    def copy(self):
        return AutocarObservation(self.obs)




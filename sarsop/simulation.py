import json
import numpy as np
from utils import *
from config_utils import load_configs

config = load_configs()

NY = config["ny"]
NDX = config["ndx"]
NCARS = config["ncars"]
SUBLANE_WIDTH = config["sublane_width"]
CELL_LENGTH = config["cell_length"]
DT = config["dt"]
SUBLANES_SPEEDS = config["sublanes_speeds"]
X0_START = config["autonomous_car_start_pos"]

class Simulation:

    def __init__(self, poses_path, config_path):
        self.poses = json.load(open(poses_path, 'r'))
        self.x0 = X0_START
        self.y0 = 0
        self.t_i = 0
        self.cmds = [[self.get_vel(), 0.0, 0.0]]

    def get_vel(self):
        lane = int(round(abs(self.y0)/SUBLANE_WIDTH))
        vel = SUBLANES_SPEEDS[lane]
        return vel
    
    def step(self, action):

        self.t_i += 1

        vx = self.get_vel()
        self.x0 += vx * DT

        if action == "left":
            self.y0 = max(0, self.y0+SUBLANE_WIDTH)
            vy = SUBLANE_WIDTH / DT
        elif action == "right":
            self.y0 = min((NY-1)*SUBLANE_WIDTH, self.y0-SUBLANE_WIDTH)
            vy = -SUBLANE_WIDTH / DT
        else:
            vy = 0.0

        self.cmds.append([vx, vy, self.t_i * DT])


    def observe(self):

        o_dx = np.zeros(NCARS-1, np.int32)

        for car in range(1,NCARS):

            x_car = self.poses["robot_%d" % car][self.t_i][0]
            dx = x_car - self.x0

            dx_discretized = int(round(dx / CELL_LENGTH))
            if dx_discretized >= 1:
                dx_i  = NDX-1
            elif dx_discretized <= -(NDX-2):
                dx_i = 0
            else:
                dx_i = dx_discretized + NDX-2

            print("True dx_i :", dx_i)

            probas = make_dx_obs_matrix()
            probas_car = probas[dx_i,:]
            o_dx[car-1] = np.random.choice(np.arange(NDX), p=probas_car)

        return o_dx

    def write_commands(self):
        with open("cmds.json", "w") as f:
            json.dump(self.cmds, f)

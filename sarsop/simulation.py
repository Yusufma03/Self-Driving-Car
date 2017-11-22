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
LANES_SPEEDS = config["lanes_speeds"]

class Simulation:

    def __init__(self, poses_path, config_path):
        self.poses = json.load(open(poses_path, 'r'))
        self.x0 = X0_START
        self.y0 = 0
        self.t_i = 0
        self.cmds = [[0.0, 0.0, 0.0]]

    
    def step(self, action):
        lateral_action,speed_action_id = action
        speed = LANES_SPEEDS[speed_action_id]

        self.t_i += 1

        vx = speed
        self.x0 += vx * DT

        if lateral_action == "left":
            self.y0 = max(0, self.y0+SUBLANE_WIDTH)
            vy = SUBLANE_WIDTH / DT
        elif lateral_action == "right":
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

            lookahead = int(NDX/2)
            dx_discretized = int(round(dx / CELL_LENGTH))
            if dx_discretized >= lookahead:
                dx_i  = NDX-1
            elif dx_discretized <= -lookahead:
                dx_i = 0
            else:
                dx_i = lookahead + dx_discretized

            probas = make_dx_obs_matrix()
            probas_car = probas[dx_i,:]
            o_dx[car-1] = np.random.choice(np.arange(NDX), p=probas_car)

        return o_dx

    def write_commands(self):
        with open("cmds.json", "w") as f:
            json.dump(self.cmds, f)

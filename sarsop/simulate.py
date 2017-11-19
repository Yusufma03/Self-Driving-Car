import xml.etree.ElementTree as ET
import sys
import json
import numpy as np
from utils import *

NY = 4
NDX = 5
NCARS = 2
NB_TIMESTEPS = 300
LANE_OFFSET_Y = 3.0
FIRST_LANE_Y = 0
CELL_SIZE_X = 2.0

DT = 0.1

def get_sublanes_speeds():
    # TODO
    return [20, 20, 40, 40]

def parse_policy(path):

    tree = ET.parse(path)
    root = tree.getroot()

    alpha_vector = root.find("AlphaVector")
    vec_len = int(alpha_vector.attrib["vectorLength"])
    num_vecs = int(alpha_vector.attrib["numVectors"])

    vectors = np.zeros((num_vecs, vec_len))
    actions_vectors = np.zeros(num_vecs, np.int32)


    for i,vec in enumerate(alpha_vector.findall("Vector")):
        vals = [float(x) for x in vec.text.split(" ")[:-1]]
        vectors[i,:] = vals
        actions_vectors[i] = int(vec.attrib["action"])

    return vectors,actions_vectors


def get_optimal_action(belief, vectors, actions_vectors):

    b_y0,b_dx = belief
    b_y0 = b_y0.reshape(-1,1)
    b_dx = b_dx.reshape(1,-1)

    b = b_y0.dot( b_dx ).flatten()
    vals = vectors.dot( b )

    ind = np.argmax(vals)
    action_id = actions_vectors[ind]

    return ["none", "left", "right"][action_id]


def action_update(belief, action):

    b_y0_0,b_dx_0 = belief

    y0_trans = make_y0_transition_matrix(action)
    b_y0_1 = y0_trans.T.dot( b_y0_0 )

    b_dx_1 = np.zeros((NCARS-1, NDX))
    for car in range(1,NCARS):
        for y0 in range(NY):

            dx_trans = make_dx_transition_matrix(y0, car)
            b_dx_1[car-1,:] += b_y0_0[y0] * dx_trans.T.dot( b_dx_0[car-1,:] )

    return b_y0_1,b_dx_1

def observation_update(belief, obs):

    b_y0_0,b_dx_0 = belief

    b_y0_1 = b_y0_0

    # obs : (NCARS-1)
    b_dx_1 = np.zeros((NCARS-1, NDX))
    for car in range(1,NCARS):
        obs_mat = make_dx_obs_matrix()
        b_dx_1[car-1,:] = obs_mat[:,obs[car-1]] * b_dx_0[car-1,:]
        b_dx_1[car-1,:] /= np.sum(b_dx_1[car-1,:])

    return b_y0_1,b_dx_1


class Simulation:

    def __init__(self, poses_path, config_path):
        self.poses = json.load(open(poses_path, 'r'))
        self.config = json.load(open(config_path, 'r'))
        self.x0 = self.config["autonomous_car_start_pos"]
        self.y0 = FIRST_LANE_Y
        self.t_i = 0
        self.sublanes_speeds = get_sublanes_speeds()
    
    def step(self, action):

        self.t_i += 1

        def get_vel(y0):
            lane = int(round((self.y0 - FIRST_LANE_Y)/LANE_OFFSET_Y))
            vel = self.sublanes_speeds[lane]
            return vel

        v = get_vel(self.y0)
        self.x0 += v * DT

        if action == "left":
            self.y0 = max(0, self.y0-LANE_OFFSET_Y)
        elif action == "right":
            self.y0 = min((NY-1)*LANE_OFFSET_Y, self.y0+LANE_OFFSET_Y)


    def observe(self):

        o_dx = np.zeros(NCARS-1, np.int32)

        for car in range(1,NCARS):

            x_car = self.poses["robot_%d" % car][self.t_i][0]
            dx = x_car - self.x0
            dx_i = int(round(dx / CELL_SIZE_X))

            probas = make_dx_obs_matrix()
            probas_car = probas[car-1,:]
            o_dx[car-1] = np.random.choice(np.arange(NDX), p=probas_car)

        return o_dx



vectors,actions_vectors = parse_policy(sys.argv[1])

simulation = Simulation("poses.json", "lane_config.json")

# Initial belief
b_y0 = np.zeros(NY)
b_y0[0] = 1.0
b_dx = np.ones((NCARS-1, NDX)) * 0.5
belief = b_y0,b_dx

#for i in range(NB_TIMESTEPS):
for i in range(5):

    action = get_optimal_action(belief, vectors, actions_vectors)
    simulation.step(action)
    belief = action_update(belief, action)
    obs = simulation.observe()
    belief = observation_update(belief, obs)

    print(action)
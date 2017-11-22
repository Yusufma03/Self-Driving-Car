# simulate_policy.py
# Takes a policy file as input and tests it against a simulation.
# The simulation uses recorded car positions created by the script "generate_poses.py"

import xml.etree.ElementTree as ET
import sys
import json
import numpy as np
from utils import *
from config_utils import load_configs
from simulation import Simulation

config = load_configs()

NY = config["ny"]
NDX = config["ndx"]
NCARS = config["ncars"]
NB_TIMESTEPS = config["nb_timesteps"]
SUBLANE_WIDTH = config["sublane_width"]
DT = config["dt"]
LANES_SPEEDS_CELLS = config["lanes_speeds_cells"]



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

    b = b_y0.reshape(-1,1)
    for i in range(b_dx.shape[0]):
        b = b.dot( b_dx[i,:].reshape(1,-1) ).reshape(-1,1)

    vals = vectors.dot( b )

    ind = np.argmax(vals)
    action_id = actions_vectors[ind]

    nb_speeds = len(LANES_SPEEDS_CELLS)

    lateral_action_id = action_id // nb_speeds
    speed_action_id = action_id % nb_speeds

    return ["none", "left", "right"][lateral_action_id],speed_action_id


def action_update(belief, action):

    b_y0_0,b_dx_0 = belief
    lateral_action,speed_action_id = action
    speed = LANES_SPEEDS_CELLS[speed_action_id]

    y0_trans = make_y0_transition_matrix(lateral_action)
    b_y0_1 = y0_trans.T.dot( b_y0_0 )

    b_dx_1 = np.zeros((NCARS-1, NDX))
    for car in range(1,NCARS):
        dx_trans = make_dx_transition_matrix(speed, car)
        b_dx_1[car-1,:] = dx_trans.T.dot( b_dx_0[car-1,:] )

    return b_y0_1,b_dx_1

def observation_update(belief, obs):

    b_y0_0,b_dx_0 = belief

    b_y0_1 = b_y0_0

    b_dx_1 = np.zeros((NCARS-1, NDX))
    for car in range(1,NCARS):
        obs_mat = make_dx_obs_matrix()
        b_dx_1[car-1,:] = obs_mat[:,obs[car-1]] * b_dx_0[car-1,:]
        b_dx_1[car-1,:] /= np.sum(b_dx_1[car-1,:])

    return b_y0_1,b_dx_1

vectors,actions_vectors = parse_policy(sys.argv[1])
simulation = Simulation("poses.json", "lane_config.json")

# Initial belief
b_y0 = np.zeros(NY)
b_y0[0] = 1.0
b_dx = np.ones((NCARS-1, NDX)) / NDX
belief = b_y0,b_dx

for i in range(NB_TIMESTEPS-1):

    print("Iter %d ----------" % i)
    print("Initial belief :",belief[1])
    action = get_optimal_action(belief, vectors, actions_vectors)
    print("Action :",action)
    simulation.step(action)
    belief = action_update(belief, action)
    print("After action update :",belief[1])
    obs = simulation.observe()
    print("Observation :",obs)
    belief = observation_update(belief, obs)
    print("After observation update :",belief[1])

    print("\n")

simulation.step(('none',1))
simulation.write_commands()

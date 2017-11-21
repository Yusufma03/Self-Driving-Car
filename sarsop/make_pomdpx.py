import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import sys
from utils import *
from config_utils import load_configs

config = load_configs()

NY = config["ny"]
NDX = config["ndx"]
NCARS = config["ncars"]
CARS_SUBLANES = config["cars_sublanes"]
DISCOUNT = config["discount"]
REWARDS = config["rewards"]

NVX = 2
SPEEDS = [1,2]
SPEED_OBS_PROBAS = [ [0.8, 0.2], [0.2, 0.8] ]


pomdpx = ET.Element("pomdpx")

# ------------ Discount ----------------

discount = ET.SubElement(pomdpx, "Discount")
discount.text = str(DISCOUNT)


# ------------ Variable -----------------

variable = ET.SubElement(pomdpx, "Variable")

#### State variables ####
def make_state_var(varname, numvals, fullyobs=False):
    statevar = ET.SubElement(variable, "StateVar", vnamePrev=varname + "_0", vnameCurr=varname + "_1", fullyObs=str(fullyobs))
    numvalues = ET.SubElement(statevar, "NumValues")
    numvalues.text = str(numvals)

make_state_var("y0", NY, fullyobs=True)

for i in range(1, NCARS):
    make_state_var("dx%d" % i, NDX, fullyobs=False)
    make_state_var("vx%d" % i, NVX, fullyobs=False)

#### Observation variables ####
for i in range(1, NCARS):
    obsvar = ET.SubElement(variable, "ObsVar", vname="o_dx%d" % i)
    ET.SubElement(obsvar, "NumValues").text = str(NDX)
    
    obsvar = ET.SubElement(variable, "ObsVar", vname="o_vx%d" % i)
    ET.SubElement(obsvar, "NumValues").text = str(NVX)

#### Action variables ####
actionvar = ET.SubElement(variable, "ActionVar", vname="action")
valueenum = ET.SubElement(actionvar, "ValueEnum")
valueenum.text = "none left right"

#### Reward variables ####
rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_obj")
rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_out")
rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_lane")
for i in range(1,NCARS):
    rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_collision%d" % i)


# ------------ InitialStateBelief ----------------

initialstatebelief = ET.SubElement(pomdpx, "InitialStateBelief")

def make_initial_cond_prob(varname, prob_str):
    condprob = ET.SubElement(initialstatebelief, "CondProb")
    ET.SubElement(condprob, "Var").text = varname
    ET.SubElement(condprob, "Parent").text = "null"
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")
    entry = ET.SubElement(parameter, "Entry")

    ET.SubElement(entry, "Instance").text = "-"
    ET.SubElement(entry, "ProbTable").text = prob_str

p = np.zeros(NY)
p[0] = 1.0
make_initial_cond_prob("y0_0", table_to_str(p)) 

for i in range(1, NCARS):
    make_initial_cond_prob("dx%d_0" % i, "uniform")
    make_initial_cond_prob("vx%d_0" % i, "uniform")


# ------------ StateTransitionFunction ----------------

def make_transition_dx_lane_others(n):
    condprob = ET.SubElement(statetransitionfunction, "CondProb")
    ET.SubElement(condprob, "Var").text = "dx%d_1" % n
    ET.SubElement(condprob, "Parent").text = "y0_0 vx%d_0 dx%d_0" % (n,n)
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")

    for i in range(NY):
        for j in range(NVX):

            entry = ET.SubElement(parameter, "Entry")
            ET.SubElement(entry, "Instance").text = "s%d s%d - -" % (i,j)
            p = make_dx_transition_matrix(i, SPEEDS[j])
            ET.SubElement(entry, "ProbTable").text = table_to_str(p)

def make_transition_vx_lane_others(n):
    condprob = ET.SubElement(statetransitionfunction, "CondProb")
    ET.SubElement(condprob, "Var").text = "vx%d_1" % n
    ET.SubElement(condprob, "Parent").text = "vx%d_0" % n
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")

    entry = ET.SubElement(parameter, "Entry")
    ET.SubElement(entry, "Instance").text = "- -"
    p = make_vx_transition_matrix(n)
    ET.SubElement(entry, "ProbTable").text = table_to_str(p)


statetransitionfunction = ET.SubElement(pomdpx, "StateTransitionFunction")

#### y0_1 ####

condprob = ET.SubElement(statetransitionfunction, "CondProb")
ET.SubElement(condprob, "Var").text = "y0_1"
ET.SubElement(condprob, "Parent").text = "action y0_0"
parameter = ET.SubElement(condprob, "Parameter", type="TBL")

# "None" action
entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "none - -"
ET.SubElement(entry, "ProbTable").text = "identity"

# "Left" action
entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "left - -"
p = make_y0_transition_matrix("left")
ET.SubElement(entry, "ProbTable").text = table_to_str(p) 

# "Right" action
entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "right - -"
p = make_y0_transition_matrix("right")
ET.SubElement(entry, "ProbTable").text = table_to_str(p) 


#### dxi_1 ####
for i in range(1, NCARS):
    make_transition_dx_lane_others(i) 

#### vxi_1 ####
for i in range(1, NCARS):
    make_transition_vx_lane_others(i)



# ------------ ObsFunction ----------------

def make_obs_cond_prob(varname, n, str_probtable):
    
    condprob = ET.SubElement(obsfunction, "CondProb")
    ET.SubElement(condprob, "Var").text = "o_%s%d" % (varname, n)
    ET.SubElement(condprob, "Parent").text = "%s%d_1" % (varname, n)
    parameter = ET.SubElement(condprob, "Parameter")
    entry = ET.SubElement(parameter, "Entry")
    ET.SubElement(entry, "Instance").text = "- -"
    ET.SubElement(entry, "ProbTable").text = str_probtable


obsfunction = ET.SubElement(pomdpx, "ObsFunction")

for i in range(1, NCARS):

    #### o_xi ####
    p = make_dx_obs_matrix()
    p_str = table_to_str(p)
    make_obs_cond_prob("dx", i, p_str)
    
    #### o_vxi ####
    p = np.array(SPEED_OBS_PROBAS)
    p_str = table_to_str(p)
    make_obs_cond_prob("vx", i, p_str)


# ------------ RewardFunction ----------------

rewardfunction = ET.SubElement(pomdpx, "RewardFunction")

def make_reward(name, parents, v_str):

    func = ET.SubElement(rewardfunction, "Func")
    ET.SubElement(func, "Var").text = name
    ET.SubElement(func, "Parent").text = " ".join(parents)
    parameter = ET.SubElement(func, "Parameter", type="TBL")

    entry = ET.SubElement(parameter, "Entry")
    ET.SubElement(entry, "Instance").text = " ".join("-" for i in range(len(parents)))
    ET.SubElement(entry, "ValueTable").text = v_str


# Objective reward
v = np.zeros(NY, np.int32)
v[-1] = REWARDS["objective"]
make_reward("rew_obj", ["y0_1"], table_to_str(v, mode='int'))

# Lane change reward
v = np.zeros(3, np.int32)
v[1:] = REWARDS["lane_change"]
make_reward("rew_lane", ["action"], table_to_str(v, mode='int'))

# Out of lane reward
v = np.zeros((3,NY), np.int32)
v[1,0] = REWARDS["out_of_lane"]
v[2,-1] = REWARDS["out_of_lane"]
make_reward("rew_out", ["action", "y0_1"], table_to_str(v, mode='int'))

# Collision reward
for i in range(1, NCARS):
    v = np.zeros((NY,NDX), np.int32)
    lane = CARS_SUBLANES[i]
    v[lane, -2] = REWARDS["collision"]
    make_reward("rew_collision%d" % i, ["y0_1", "dx%d_1" % i], table_to_str(v, mode='int'))

# ------------ Save file ----------------

raw_str = ET.tostring(pomdpx, 'utf-8')
reparsed = minidom.parseString(raw_str)
pretty_str = reparsed.toprettyxml()

with open("autogenerated.pomdpx", "w") as f:
    f.write(pretty_str)

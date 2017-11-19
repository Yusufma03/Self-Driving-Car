import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import sys
from utils import *

NY = 4
NDX = 5
NCARS = 2
CARS_LANES = [0, 3]
DISCOUNT = 0.95
UNCERTAINTIES_PROBAS_X = [0.2, 0.6, 0.2]
SUBLANES_SPEEDS = [1,1,2,2]
COLLISION_REWARD = -1000
LEFT_REWARD = -1
RIGHT_REWARD = 1
OUT_OF_LANE_REWARD = -10


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

#### Observation variables ####
for i in range(1, NCARS):
    obsvar = ET.SubElement(variable, "ObsVar", vname="o_dx%d" % i)
    ET.SubElement(obsvar, "NumValues").text = str(NDX)

#### Action variables ####
actionvar = ET.SubElement(variable, "ActionVar", vname="action")
valueenum = ET.SubElement(actionvar, "ValueEnum")
valueenum.text = "none left right"

#### Reward variables ####
rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_action")


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


# ------------ StateTransitionFunction ----------------

def make_transition_x_lane_others(n):
    condprob = ET.SubElement(statetransitionfunction, "CondProb")
    ET.SubElement(condprob, "Var").text = "dx%d_1" % n
    ET.SubElement(condprob, "Parent").text = "y0_0 dx%d_0" % n
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")

    for i in range(NY):

        entry = ET.SubElement(parameter, "Entry")
        ET.SubElement(entry, "Instance").text = "s%d - -" % i
        p = make_dx_transition_matrix(i, n)
        print(p)
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
    make_transition_x_lane_others(i) 


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


# ------------ RewardFunction ----------------

rewardfunction = ET.SubElement(pomdpx, "RewardFunction")

func = ET.SubElement(rewardfunction, "Func")
ET.SubElement(func, "Var").text = "rew_action"
ET.SubElement(func, "Parent").text = "action y0_1"
parameter = ET.SubElement(func, "Parameter", type="TBL")

entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "none -"
v = np.zeros(NY, np.int32)
ET.SubElement(entry, "ValueTable").text = table_to_str(v, mode='int')

entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "left -"
v = np.ones(NY, np.int32) * LEFT_REWARD
v[0] = OUT_OF_LANE_REWARD
ET.SubElement(entry, "ValueTable").text = table_to_str(v, mode='int')

entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "right -"
v = np.ones(NY, np.int32) * RIGHT_REWARD
v[-1] = OUT_OF_LANE_REWARD
ET.SubElement(entry, "ValueTable").text = table_to_str(v, mode='int')

# ------------ Save file ----------------

raw_str = ET.tostring(pomdpx, 'utf-8')
reparsed = minidom.parseString(raw_str)
pretty_str = reparsed.toprettyxml()

with open("test_autogenerated.pomdpx", "w") as f:
    f.write(pretty_str)

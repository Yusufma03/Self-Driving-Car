import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np

NY = 4
NX = 100
NCARS = 2
DISCOUNT = 0.95

A_CAR_INIT_CELL = (1, 0)
O_CARS_INIT_CELLS = [(1, 3)]
UNCERTAINTIES_PROBAS_X = [0.2, 0.6, 0.2]
SUBLANES_SPEEDS = [1,1,2,2]
GOAL_REWARD = 5
COLLISION_REWARD = -1000
LEFT_REWARD = -1
RIGHT_REWARD = -1

def table_to_str(table):

    if len(table.shape) == 1:
        table = table.reshape(1,-1)
    
    s = ""

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            s += "%.1f" % table[i,j]
            if i != table.shape[0]-1 or j != table.shape[1]-1:
                s += ' '

    return s

pomdpx = ET.Element("pomdpx")

# ------------ Discount ----------------

discount = ET.SubElement(pomdpx, "Discount")
discount.text = str(DISCOUNT)


# ------------ Variable -----------------

variable = ET.SubElement(pomdpx, "Variable")

#### State variables ####
def make_state_var(varname, n, numvals, fullyobs=False):
    if fullyobs:
        statevar = ET.SubElement(variable, "StateVar", vnamePrev=("%s%d_0" % (varname, n)), vnameCurr=("%s%d_1" % (varname, n)), fullyObs="true")
    else:
        statevar = ET.SubElement(variable, "StateVar", vnamePrev=("%s%d_0" % (varname, n)), vnameCurr=("%s%d_1" % (varname, n)))
    numvalues = ET.SubElement(statevar, "NumValues")
    numvalues.text = str(numvals)

make_state_var("x", 0, NX, fullyobs=True)
make_state_var("y", 0, NY, fullyobs=True)

for i in range(1, NCARS):
    make_state_var("x", i, NX)
    make_state_var("y", i, NY)

#### Observation variables ####
for i in range(1, NCARS):
    obsvar = ET.SubElement(variable, "ObsVar", vname="o_x%d" % i)
    ET.SubElement(obsvar, "NumValues").text = str(NX)
    obsvar = ET.SubElement(variable, "ObsVar", vname="o_y%d" % i)
    ET.SubElement(obsvar, "NumValues").text = str(NY)

#### Action variables ####
actionvar = ET.SubElement(variable, "ActionVar", vname="action")
valueenum = ET.SubElement(actionvar, "ValueEnum")
valueenum.text = "none left right"

#### Reward variables ####
rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_goal")
rewardvar = ET.SubElement(variable, "RewardVar", vname="rew_action")


# ------------ InitialStateBelief ----------------

initialstatebelief = ET.SubElement(pomdpx, "InitialStateBelief")

def make_initial_cond_prob(varname, n, num_values, init_val, probas_obs):
    condprob = ET.SubElement(initialstatebelief, "CondProb")
    ET.SubElement(condprob, "Var").text = "%s%d_0" % (varname, n)
    ET.SubElement(condprob, "Parent").text = "null"
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")
    entry = ET.SubElement(parameter, "Entry")

    ET.SubElement(entry, "Instance").text = "-"

    probas = np.zeros(num_values)

    if probas_obs is None:
        probas[init_val] = 1.0
    else:
        middle_proba_obs = int(len(probas_obs)/2)
        probas[init_val-middle_proba_obs:init_val+middle_proba_obs+1] = probas_obs
    prob_str = table_to_str(probas)
    ET.SubElement(entry, "ProbTable").text = prob_str

make_initial_cond_prob("x", 0, NX, A_CAR_INIT_CELL[0], None)
make_initial_cond_prob("y", 0, NY, A_CAR_INIT_CELL[1], None)

for i in range(1, NCARS):
    make_initial_cond_prob("x", i, NX, O_CARS_INIT_CELLS[i-1][0], UNCERTAINTIES_PROBAS_X)
    make_initial_cond_prob("y", i, NY, O_CARS_INIT_CELLS[i-1][1], None)


# ------------ StateTransitionFunction ----------------

def make_transition_x_lane(n):
    condprob = ET.SubElement(statetransitionfunction, "CondProb")
    ET.SubElement(condprob, "Var").text = "x%d_1" % n
    ET.SubElement(condprob, "Parent").text = "y%d_0 x%d_0" % (n,n)
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")

    for y in range(NY):
        entry = ET.SubElement(parameter, "Entry")
        ET.SubElement(entry, "Instance").text = "s%d - -" % y
        p = np.identity(NX)
        speed = SUBLANES_SPEEDS[y]
        p[:-speed, :] = np.roll(p[:-speed, :], speed, axis=1)
        p[-speed:,:] = 0
        p[-speed:,-1] = 1.0
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
p = np.identity(NY)
p[1:,:] = np.roll(p[1:,:], -1, axis=1)
ET.SubElement(entry, "ProbTable").text = table_to_str(p) 

# "Right" action
entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "right - -"
p = np.identity(NY)
p[:-1,:] = np.roll(p[:-1,:], 1, axis=1)
ET.SubElement(entry, "ProbTable").text = table_to_str(p) 

#### x0_1 ####
make_transition_x_lane(0)

### Other cars ###
for i in range(1, NCARS):

    #### yi_1 ####
    condprob = ET.SubElement(statetransitionfunction, "CondProb")
    ET.SubElement(condprob, "Var").text = "y%d_1" % i
    ET.SubElement(condprob, "Parent").text = "y%d_0" % i
    parameter = ET.SubElement(condprob, "Parameter", type="TBL")
    entry = ET.SubElement(parameter, "Entry")
    ET.SubElement(entry, "Instance").text = "- -"
    ET.SubElement(entry, "ProbTable").text = "identity"

    ### xi_1 ###
    make_transition_x_lane(i)


# ------------ ObsFunction ----------------

def make_obs_x_table():
    p = np.zeros((NX,NX))
    l = len(UNCERTAINTIES_PROBAS_X)
    half_l = int(len(UNCERTAINTIES_PROBAS_X)/2)

    for i in range(p.shape[0]):
        expanded = np.zeros(NX + 2*half_l)
        expanded[i:i+l] = UNCERTAINTIES_PROBAS_X
        squeezed = expanded[half_l:half_l+NX]
        front_cut = expanded[:half_l]
        back_cut = expanded[half_l+NX:]
        squeezed[0] += np.sum(front_cut)
        squeezed[-1] += np.sum(back_cut)

        p[i,:] = squeezed

    return p

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
    p = make_obs_x_table()
    p_str = table_to_str(p)
    make_obs_cond_prob("x", i, p_str)

    #### o_yi ####
    p = np.identity(NY)
    p_str = table_to_str(p)
    make_obs_cond_prob("y", i, p_str) 


# ------------ RewardFunction ----------------

rewardfunction = ET.SubElement(pomdpx, "RewardFunction")

func = ET.SubElement(rewardfunction, "Func")
ET.SubElement(func, "Var").text = "rew_goal"
ET.SubElement(func, "Parent").text = "x0_0"
parameter = ET.SubElement(func, "Parameter", type="TBL")
entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "-"
v = np.zeros(NX)
v[-1] = GOAL_REWARD
ET.SubElement(entry, "ValueTable").text = table_to_str(v)

func = ET.SubElement(rewardfunction, "Func")
ET.SubElement(func, "Var").text = "rew_action"
ET.SubElement(func, "Parent").text = "action"
parameter = ET.SubElement(func, "Parameter", type="TBL")

entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "none"
ET.SubElement(entry, "ValueTable").text = "0"

entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "left"
ET.SubElement(entry, "ValueTable").text = str(LEFT_REWARD)

entry = ET.SubElement(parameter, "Entry")
ET.SubElement(entry, "Instance").text = "right"
ET.SubElement(entry, "ValueTable").text = str(RIGHT_REWARD)

# ------------ Save file ----------------

raw_str = ET.tostring(pomdpx, 'utf-8')
reparsed = minidom.parseString(raw_str)
pretty_str = reparsed.toprettyxml()

with open("test_autogenerated.pomdpx", "w") as f:
    f.write(pretty_str)

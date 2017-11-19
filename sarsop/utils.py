import numpy as np

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


def table_to_str(table, mode='float'):

    if len(table.shape) == 1:
        table = table.reshape(1,-1)
    
    s = ""

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if mode == 'float':
                s += "%.1f" % table[i,j]
            else:
                s += "%d" % table[i,j]
            if i != table.shape[0]-1 or j != table.shape[1]-1:
                s += ' '

    return s

def make_y0_transition_matrix(action):

    assert action in ["none", "left", "right"]

    p = np.identity(NY)
    if action == "left":
        p[1:,:] = np.roll(p[1:,:], -1, axis=1)
    elif action == "right":
        p[:-1,:] = np.roll(p[:-1,:], 1, axis=1)

    return p

def make_dx_transition_matrix(y0, car):

    assert y0 >= 0 and y0 < NY
    assert car >= 1 and car < NCARS

    speed = SUBLANES_SPEEDS[CARS_LANES[car]] - SUBLANES_SPEEDS[y0]

    p = np.identity(NDX)

    if speed > 0:
        p[:-speed, :] = np.roll(p[:-speed, :], speed, axis=1)
        p[-speed:,:] = 0
        p[-speed:,-1] = 1.0

        p[0,:] = 0
        p[0,0] = 0.5
        p[0,1] = 0.5

    elif speed < 0:
        p[-speed:, :] = np.roll(p[-speed:, :], speed, axis=1)
        p[:-speed,:] = 0
        p[:-speed,0] = 1.0

    return p

def make_dx_obs_matrix():
    p = np.zeros((NDX,NDX))
    l = len(UNCERTAINTIES_PROBAS_X)
    half_l = int(len(UNCERTAINTIES_PROBAS_X)/2)

    for i in range(p.shape[0]):
        expanded = np.zeros(NDX + 2*half_l)
        expanded[i:i+l] = UNCERTAINTIES_PROBAS_X
        squeezed = expanded[half_l:half_l+NDX]
        front_cut = expanded[:half_l]
        back_cut = expanded[half_l+NDX:]
        squeezed[0] += np.sum(front_cut)
        squeezed[-1] += np.sum(back_cut)

        p[i,:] = squeezed

    return p

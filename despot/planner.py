from despot import *

def load_data():
    with open('poses.json', 'r') as fin:
        parsed = json.load(fin)
    return parsed

def get_agent_poses(dic, index):
    ret = [
        v[index][:2]
        for k, v in dic.items()
    ]
    return ret

def x2belief(x):
    belief = np.zeros(ROAD_LEN)
    x = int(x)
    belief[x] = 0.8
    belief[x-1] = 0.1
    belief[x+1] = 0.1
    return belief

def to_belief(agent_poses):
    beliefs = []
    for pos in agent_poses:
        x, y = pos
        x_belief = x2belief(x)
        beliefs.append([x_belief, y])
    return beliefs

def shift_belief(belief, bits):
    if bits > 0:
        return np.concatenate((np.zeros(bits), belief[:-bits]))
    else:
        return np.concatenate((belief[bits:], np.zeros(bits)))

def update_belief(beliefs, observations):
    new_beliefs = []
    for i in range(len(beliefs)):
        belief = beliefs[i]
        x_belief, y = belief
        if y < BOUNDARY:
            vel_x = 4
        else:
            vel_x = 2
        x_belief_shifted = shift_belief(x_belief, vel_x)
        x_obs, y_obs = observations[i]
        x_belief_obs = x2belief(x_obs)
        x_belief_new = x_belief_shifted * x_belief_obs
        x_belief_new = x_belief_new / np.sum(x_belief_new)
        new_beliefs.append([x_belief_new, y_obs])
    return new_beliefs

def action2vel(action, robot_pos):
    x, y = robot_pos
    if y < BOUNDARY:
        vel_x = 4
    else:
        vel_x = 2
    if action == LEFT and y > LEFT_MOST:
        vel_y = -1
    elif action == RIGHT and y < RIGHT_MOST:
        vel_y = +1
    else:
        vel_y = 0

    return [vel_x, vel_y]
        

if __name__=='__main__':
    data = load_data()
    index = 0
    robot_pos = [28, 0]
    dump = []
    while robot_pos[0] < 150:
        agent_poses = get_agent_poses(data, index)
        if index == 0:
            agent_belief = to_belief(agent_poses)
        else:
            agent_belief = update_belief(agent_belief, agent_poses)

        despot_tree = build_despot(robot_pos, agent_belief)
        action = planning(despot_tree)
        print(action)
        vel_x, vel_y = action2vel(action, robot_pos)
        robot_pos = [robot_pos[0] + vel_x, robot_pos[1] + vel_y]
        print(robot_pos)
        dump.append([vel_x*10, vel_y*10, index / 10.0])
        index += 1
    
    with open('cmds.json', 'w') as fout:
        json.dump(dump, fout)
from despot import *
import pickle
import time

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
    belief = np.zeros(ROAD_LEN+100)
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

def update_belief(beliefs, observations, old_observations):
    new_beliefs = []
    for i in range(len(beliefs)):
        belief = beliefs[i]
        x_belief, y = belief
        x_obs, y_obs = observations[i]
        x_obs_old, y_obs_old = old_observations[i]
        vel_x = int(x_obs - x_obs_old)
        x_belief_shifted = shift_belief(x_belief, vel_x)
        x_belief_obs = x2belief(x_obs)
        x_belief_new = x_belief_shifted * x_belief_obs
        x_belief_new = x_belief_new / np.sum(x_belief_new)
        new_beliefs.append([x_belief_new, y_obs])
    return new_beliefs

def action2vel(action, robot_pos):
    x, y = robot_pos
    if y < BOUNDARY:
        vel_x = 4
        if action in [LEFT, STAY, RIGHT]:
            vel_x = 2
    else:
        vel_x = 2
    if (action == LEFT or action == LEFT_FAST) and y > LEFT_MOST+1:
        vel_y = -1
    elif (action == RIGHT or action == RIGHT_FAST) and y < RIGHT_MOST:
        vel_y = +1
    else:
        vel_y = 0

    return [vel_x, vel_y]

def load_config(config):
    global SEARCH_DEPTH
    global GAMMA
    global GOAL
    global EPSILON
    global NUM_OF_SCENARIOS
    global TIME_LEN
    global LAMBDA
    global ROAD_LEN
    
    SEARCH_DEPTH = config['search_depth']
    GAMMA = config['gamma']
    GOAL = config['goal']
    EPSILON = config['epsilon']
    NUM_OF_SCENARIOS = config['num_of_scenarios']
    TIME_LEN = config['time_len']
    LAMBDA = config['lambda']
    ROAD_LEN = config['road_len']

def change_config(config, scenarios, lbd, epsilon):
    config['num_of_scenarios'] = scenarios
    config['lambda'] = lbd
    config['epsilon'] = epsilon
    return config
        

def main(scenarios, lbd, epsilon):
    data = load_data()
    index = 0
    with open('despot_config.json', 'r') as fin:
        config = json.load(fin)
    config = change_config(config,scenarios, lbd, epsilon)
    load_config(config)
    with open('../ros-lanechanging/autocar/scripts/lane_config.json', 'r') as fin:
        parsed = json.load(fin)
        robot_start_x = parsed['autonomous_car_start_pos']
    robot_pos = [robot_start_x, 0]
    dump = []
    despot_tree_size = 0
    cnt = 0
    data_len = len(data['robot_1'])
    while index < data_len and robot_pos[0] < GOAL-5:
        agent_poses_new = get_agent_poses(data, index)
        if index == 0:
            agent_belief = to_belief(agent_poses_new)
        else:
            agent_belief = update_belief(agent_belief, agent_poses_new, agent_poses_old)

        despot_tree = build_despot(robot_pos, agent_belief, NUM_OF_SCENARIOS)
        despot_tree_size += len(despot_tree)
        cnt += 1
        action = planning(despot_tree, NUM_OF_SCENARIOS)
        print(action)
        vel_x, vel_y = action2vel(action, robot_pos)
        robot_pos = [robot_pos[0] + vel_x, robot_pos[1] + vel_y]
        print(robot_pos)
        dump.append([vel_x*10, vel_y*10, index / 10.0])
        index += 1
        agent_poses_old = agent_poses_new
    
    with open('cmds.json', 'w') as fout:
        json.dump(dump, fout)

    return despot_tree_size, cnt, estimate_reward([robot_start_x, 0], data, dump)

def vel2act(vel):
    vx, vy, t = vel
    if vy == 0 and vx == 2:
        return STAY
    elif vy == 0 and vx == 4:
        return STAY_FAST
    elif vy == -1 and vx == 2:
        return LEFT
    elif vy == -1 and vx == 4:
        return LEFT_FAST
    elif vy == 1 and vx == 2:
        return RIGHT
    else:
        return RIGHT_FAST

def check_collision(robot, agents):
    for agent in agents:
        if robot[1] == agent[1] and abs(robot[0] - agent[0]) <= 2:
            return True
    return False

def estimate_reward(robot_pos, agent_pos, vels):
    vels_ = np.array(vels) / 10
    robot = robot_pos
    total_reward = 0
    cnt = 0
    for vel in vels_:
        action = vel2act(vel)
        agents = get_agent_poses(agent_pos, cnt)
        cnt += 1
        reward = 0
        if check_collision(robot, agents):
            reward += -10000
        if action in [LEFT_FAST, RIGHT_FAST, STAY_FAST]:
            reward += 5
        if robot[1] != -3:
            reward -= 10
        if robot[1] < LEFT_MOST:
            reward += -100
        if robot[1] > RIGHT_MOST:
            reward += -100
        total_reward += reward
        robot[0] += vel[0]
        robot[1] += vel[1]
    return total_reward

# if __name__ == '__main__':
#     rewards = []
#     tree_size = []
#     t = []
#     for scenarios in [5, 10, 20, 40, 80]:
#         # size_s = []
#         # cnt_s = []
#         # reward_s = []
#         # time_s = []
#         # size, cnt, reward = main(scenarios, 10, 0.5)
#         # for _ in range(5):
#         #     start = time.time()
#         #     size_t, cnt_t, reward_t = main(scenarios, 10, 0.5)
#         #     used = time.time() - start
#         #     size_s.append(size_t / cnt_t)
#         #     cnt_s.append(cnt_t)
#         #     reward_s.append(reward_t)
#         #     time_s.append(used / cnt_t)
#         # size, cnt, reward = sum(size_s) / len(size_s), sum(cnt_s) / len(cnt_s), sum(reward_s) / len(reward_s)
#         # t_used = sum(time_s) / len(time_s)
#         start = time.time()
#         size, cnt, reward = main(scenarios, 10, 0.5)
#         t_used = time.time() - start
#         print(size, reward, t_used)
#         rewards.append(reward)
#         tree_size.append(size / cnt)
#         t.append(t_used)

#     import pdb; pdb.set_trace()
    
#     with open('result.pkl', 'wb') as fout:
#         pickle.dump(rewards, fout, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(tree_size, fout, pickle.HIGHEST_PROTOCOL)
#         pickle.dump(t, fout, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    start = time.time()
    size, cnt, reward = main(40, 10, 0.5)
    used = time.time() - start
    print(size / cnt, reward, used)

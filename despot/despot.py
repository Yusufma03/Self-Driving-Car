from utils import *

def build_despot(robot_pos, agent_belief):
    nodes_array = []
    root = Node(None, 0, 0)

    for _ in range(NUM_OF_SCENARIOS):
        rand_nums = np.random.random(size=SEARCH_DEPTH+1)
        scenario = Scenario(SEARCH_DEPTH, rand_nums)
        robot = RobotCar(robot_pos)
        agents = []
        for belief in agent_belief:
            agent = AgentCar(belief)
            agent.init_pos(rand_nums[0])
            agents.append(agent)

        scenario.init_cars(robot, agents)
        root.add_scenario(scenario)

    nodes_array.append(root)
    start = time.time()
    while time.time() - start < TIME_LEN:
        history = []
        trial_id = trial(0, nodes_array, history)
        back_up(nodes_array, history)
    return nodes_array


def trial(node_id, nodes_array, history):
    history.append(node_id)
    node = nodes_array[node_id]
    if node.depth > SEARCH_DEPTH:
        return node_id
    if node.is_leaf():
        node.expand(nodes_array)
    node.init_bounds()
    next_id, weu = node.get_next_node(nodes_array)
    if weu >= 0:
        return trial(next_id, nodes_array, history)
    else:
        return next_id


def back_up_node(nodes_array, node_id):
    node = nodes_array[node_id]
    uppers = []
    lowers = []

    for action in range(NUM_OF_ACTIONS):
        first = (1.0 / len(node.scenarios)) * \
            np.sum([scenario.get_reward(action)
                    for scenario in node.scenarios])
        second_lower = GAMMA * np.sum(
            [
                len(nodes_array[cid].scenarios) /
                float(len(node.scenarios)) * nodes_array[cid].lower_bound()
                for _, cid in node.children[action].items()
            ]
        )
        lower = first + second_lower
        lowers.append(lower)

        second_higher = GAMMA * np.sum(
            [
                len(nodes_array[cid].scenarios) /
                float(len(node.scenarios)) * nodes_array[cid].upper_bound()
                for _, cid in node.children[action].items()
            ]
        )
        upper = first + second_higher
        uppers.append(upper)

    node.lowers = lowers
    node.uppers = uppers


def back_up(nodes_array, history):
    for node in nodes_array:
        node.init_bounds()
    for node_id in history[::-1]:
        back_up_node(nodes_array, node_id)


def planning(nodes_array):
    root = nodes_array[0]
    q_values = []
    for action in range(NUM_OF_ACTIONS):
        reward = root.get_average_reward(action)
        value = np.sum([
            regularized_value(child, nodes_array)
            for _, child in root.children[action].items()
        ])
        q_values.append(reward + value)

    return np.argmax(q_values)


def regularized_value(node_id, nodes_array):
    '''
    assume the default policy is staying in the current lane, hence default value is 0
    '''
    node = nodes_array[node_id]
    v1 = -LAMBDA
    if node.is_leaf():
        return v1

    v2s = []
    for action in range(NUM_OF_ACTIONS):
        v2_t = node.get_average_reward(action)
        v2_t += np.sum([
            regularized_value(child, nodes_array)
            for _, child in node.children[action].items()
        ])
        v2s.append(v2_t)

    v2 = np.max(v2s)
    return max(v1, v2)

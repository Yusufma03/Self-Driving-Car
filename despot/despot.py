import numpy as np
import pdb
import json
from copy import deepcopy
import time

SEARCH_DEPTH = 10
GAMMA = 0.9
CARS = 5
LEFT = 0
STAY = 1
RIGHT = 2
GOAL = 400
BOUNDARY = -1
EPSILON = 0.5
NUM_OF_SCENARIOS = 20
TIME_LEN = 2


class AgentCar(object):
    '''
    car moves horizontally, which means the lane is specified by te y coordinate
    consider the movement every 0.1 second
    '''

    def __init__(self, obs):
        self.obs = obs
        self.bias = np.random.random()

    def init_pos(self, rand_num):
        rand = (self.bias + rand_num) / 2
        x, y = self.obs
        self.y = y
        if rand < 0.1:
            self.x = x - 2
        elif 0.1 <= rand < 0.9:
            self.x = x
        else:
            self.x = x + 2

    def step(self, rand_num):
        rand = (self.bias + rand_num) / 2
        if self.y < BOUNDARY:
            vel_x = 4
        else:
            vel_x = 2
        new_x, new_y = self.x + vel_x, self.y

        new_car = AgentCar([new_x, new_y])
        new_car.init_pos(rand_num)
        return new_car

    def to_str(self):
        return str(self.x) + str(self.y)

    def predict(self):
        if self.y < BOUNDARY:
            vel_x = 4
        else:
            vel_x = 2
        new_x, new_y = self.x + vel_x, self.y

        return [new_x, new_y]


class RobotCar(object):
    '''
    car moves horizontally, which means the lane is specified by te y coordinate
    consider the movement every 0.1 second
    '''

    def __init__(self, loc):
        self.x, self.y = loc

    def step(self, action):
        '''
        action has three choices: LEFT, STAY, RIGHT
        '''
        if action is LEFT:
            vel_y = -1
        elif action is RIGHT:
            vel_y = +1
        else:
            vel_y = 0
        
        if self.y < BOUNDARY:
            vel_x = 4
        else:
            vel_x = 2

        new_x, new_y = self.x + vel_x, self.y + vel_y
        return RobotCar([new_x, new_y])

    def collide(self, poses):
        for pos in poses:
            if [self.x, self.y] == pos:
                return True
        return False

    def to_str(self):
        return str(self.x) + str(self.y)

class Scenario(object):
    '''
    This class implements the scenario of a despot solver. 
    '''
    def __init__(self, search_depth, rand_nums):
        self.search_depth = search_depth
        self.rand_nums = rand_nums

    def init_cars(self, robot, agents):
        self.robot = robot
        self.agents = agents

    def collide(self):
        poses = [
            [car.x, car.y]
            for car in self.agents
        ]
        return self.robot.collide(poses)

    def step(self, action):
        new_robot = self.robot.step(action)
        rand = self.rand_nums[0]
        new_rand_nums = self.rand_nums[1:]
        new_agents = []
        for agent in self.agents:
            new_agent = agent.step(rand)
            new_agents.append(new_agent)
        new_scenario = Scenario(self.search_depth-1, rand_nums=new_rand_nums)
        new_scenario.init_cars(new_robot, new_agents)
        
        return new_scenario

    def get_reward(self, action):
        new_robot = self.robot.step(action)
        new_poses = [
            agent.predict()
            for agent in self.agents
        ]

        collision = new_robot.collide(new_poses)
        if collision:
            reward = -100
        elif new_robot.x >= GOAL:
            reward = 50
        elif action is LEFT:
            reward = 10
        else:
            reward = -0.2

        return reward

    def to_str(self):
        ret = ""
        ret += self.robot.to_str()
        for agent in self.agents:
            ret += agent.to_str()

        return ret

class Node(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.scenarios = []
        self.children = [dict() for _ in range(3)]
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        self.uppers = None
        self.lowers = None
        self.weu = None

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def expand(self):
        for action in range(3):
            for scenario in self.scenarios:
                new_scenario = scenario.step(action)
                obs = new_scenario.to_str()
                if not self.children[action][obs]:
                    new_node = Node(parent=self)
                    self.children[action][obs] = new_node
                self.children[action][obs].add_scenario(new_scenario)

    def init_bounds(self):
        if self.upper_bound is None or self.lower_bound is None:
            uppers = []
            lowers = []

            for action in range(3):
                action_upper = None
                action_lower = None
                for scenario in self.scenarios:
                    reward = scenario.get_reward(action)
                    if action_upper is None or action_lower is None:
                        action_upper, action_lower = reward, reward
                    else:
                        if action_upper < reward:
                            action_upper = reward
                        if action_lower >= reward:
                            action_lower = reward
                uppers.append(action_upper)
                lowers.append(action_lower)

            self.uppers = uppers
            self.lowers = lowers

    def get_upper_action(self):
        return np.argmin(self.uppers)

    def upper_bound(self):
        return np.min(self.uppers)

    def lower_bound(self):
        return np.max(self.lowers)

    def is_leaf(self):
        for child in self.children:
            if child:
                return False
        return True

    def get_next_node(self):
        weu = None
        action = self.get_upper_action()
        node = None

        phi = len(self.scenarios)

        for k, child in self.children[action]:
            child.init_bounds()
            upper, lower = child.upper_bound(), child.lower_bound()
            phi_prim = len(child.scenarios)
            excess = upper - lower - EPSILON * GAMMA**(-child.depth)
            weu_ = phi_prim / float(phi) * excess
            if weu is None:
                weu = weu_
            else:
                if weu_ > weu:
                    weu = weu_
                    node = child
        return node, weu

            

class DESPOT(object):
    def __init__(self, num_of_cars, num_of_scenarios, search_depth):
        self.num_of_cars = num_of_cars
        self.num_of_scenarios = num_of_scenarios
        self.search_depth = search_depth

    def init_root(self, root):
        self.root = root

    def trial(self, node):
        if node.depth > self.search_depth:
            return node
        if node.is_leaf():
            node.expand()

        node.init_bounds()
        next_node, weu = node.get_next_node()
        if weu >= 0:
            return self.trial(next_node)
        else:
            return next_node

    def poses2cars(self, poses, rand_num):
        cars = []
        for pos in poses:
            agent = AgentCar(pos)
            agent.init_pos(rand_num)
            cars.append(agent)
        return cars

    def back_up(self, node):
        lowers = []
        uppers = []

        for action in range(3):
            first = (1.0 / len(node.scenarios)) * np.sum([scenario.get_reward(action) for scenario in node.scenarios])
            second_lower = GAMMA * np.sum(
                [
                    (len(next_node.children) / float(len(node.children))) * next_node.lower_bound()
                    for _, next_node in node.children[action].items()
                ]
            )
            lowers.append(first + second_lower)
            
            second_upper = GAMMA * np.sum(
                [
                    (len(next_node.children) / float(len(node.children))) *
                    next_node.upper_bound()
                    for _, next_node in node.children[action].items()
                ]
            )
            uppers.append(first + second_upper)

        lower_ret = np.max(lowers)
        upper_ret = np.min(uppers)

        node.lowers = lowers
        node.uppers = uppers

        return lower_ret, upper_ret     

    def build_despot(self, robot_loc, poses):
        for _ in range(self.num_of_scenarios):
            scenario = None
            rand_nums = np.random.random(size=self.search_depth)
            rand = rand_nums[0]
            scenario = Scenario(self.search_depth, rand_nums)
            robot = RobotCar(robot_loc)
            agents = self.poses2cars(poses, rand)
            scenario.init_cars(robot, agents)
            self.root.add_scenario(scenario)

        start = time.time()
        while time.time() - start < TIME_LEN:
            b = self.trial(self.root)
            b.init_bounds()
            self.back_up(b)
        
        return root

    
if __name__=="__main__":
    with open("./poses.json", 'r') as fin:
        parsed = json.load(fin)
    import pdb; pdb.set_trace()

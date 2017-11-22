import numpy as np
import pdb
import json
from copy import deepcopy
import time

SEARCH_DEPTH = 5
GAMMA = 0.9
CARS = 5
LEFT = 0
STAY = 1
RIGHT = 2
LEFT_FAST = 3
STAY_FAST = 4
RIGHT_FAST = 5
GOAL = 160
BOUNDARY = -1
EPSILON = 0.5
NUM_OF_SCENARIOS = 20
TIME_LEN = 1
LEFT_MOST = -4
RIGHT_MOST = 1
LAMBDA = 10
ROAD_LEN = 160
NUM_OF_ACTIONS = 6


class AgentCar(object):
    '''
    car moves horizontally, which means the lane is specified by te y coordinate
    consider the movement every 0.1 second
    '''

    def __init__(self, belief):
        self.belief = belief
        self.bias = np.random.random()

    def init_pos(self, rand_num):
        rand = (self.bias + rand_num) / 2
        x_belief, y = self.belief
        x = self.belief_choice(x_belief, rand)
        self.x, self.y = x, y

    
    def belief_choice(self, belief, rand_num):
        s = 0
        for i in range(len(belief)):
            s_new = s + belief[i]
            if s <= rand_num < s_new:
                return i
            s = s_new
        return -1

    def step(self, rand_num):
        rand = (self.bias + rand_num) / 2
        if self.y < BOUNDARY:
            vel_x = 4
            if rand < 0.2:
                vel_x = 2
        else:
            vel_x = 2
        new_x, new_y = self.x + vel_x, self.y

        new_car = deepcopy(self)
        new_car.obs = [new_x, new_y]
        new_car.init_pos(rand_num)
        return new_car

    def to_str(self):
        return str(self.x) + str(self.y)

    def predict(self):
        random = np.random.random()
        if self.y < BOUNDARY:
            vel_x = 4
            if random < 0.2:
                vel_x = 2
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
        # if action == LEFT and self.y > LEFT_MOST:
        #     vel_y = -1
        # elif action == RIGHT and self.y < RIGHT_MOST:
        #     vel_y = +1
        # else:
        #     vel_y = 0
        
        # if self.y < BOUNDARY:
        #     vel_x = 4
        # else:
        #     vel_x = 2

        if (action == LEFT or action == LEFT_FAST) and self.y > LEFT_MOST+1:
            vel_y = -1
        elif (action == RIGHT or action == RIGHT_FAST) and self.y < RIGHT_MOST:
            vel_y = +1
        else:
            vel_y = 0
        
        if self.y < BOUNDARY:
            if action in [LEFT_FAST, STAY_FAST, RIGHT_FAST]:
                vel_x = 4
            else:
                vel_x = 2
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
        reward = 0
        if collision:
            reward += -1000
        # if new_robot.x >= GOAL:
        #     reward += 50
        if action in [LEFT_FAST, STAY_FAST, RIGHT_FAST]:
            reward += 5
        if new_robot.y != -3:
            reward += -10
        if new_robot.y < LEFT_MOST:
            reward += -100
        if new_robot.y > RIGHT_MOST:
            reward += -100    

        return reward

    def to_str(self):
        ret = ""
        ret += self.robot.to_str()
        for agent in self.agents:
            ret += agent.to_str()

        return ret

class Node(object):
    def __init__(self, parent, depth, node_id):
        self.parent = parent
        self.depth = depth
        self.id = node_id
        self.scenarios = []
        self.children = [dict() for _ in range(NUM_OF_ACTIONS)]
        self.uppers = None
        self.lowers = None
        self.weu = None

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def expand(self, nodes_array):
        for action in range(NUM_OF_ACTIONS):
            for scenario in self.scenarios:
                new_scenario = scenario.step(action)
                obs = new_scenario.to_str()
                if not obs in self.children[action]:
                    new_node = Node(self.id, self.depth + 1, len(nodes_array))
                    nodes_array.append(new_node)
                    self.children[action][obs] = new_node.id
                cid = self.children[action][obs]
                nodes_array[cid].add_scenario(new_scenario)
                # self.children[action][obs].add_scenario(new_scenario)

    # def init_bounds(self):
    #     if self.uppers is None or self.lowers is None:
    #         uppers = []
    #         lowers = []

    #         for action in range(3):
    #             action_upper = None
    #             action_lower = None
    #             for scenario in self.scenarios:
    #                 reward = scenario.get_reward(action)
    #                 if action_upper is None or action_lower is None:
    #                     action_upper, action_lower = reward, reward
    #                 else:
    #                     if action_upper < reward:
    #                         action_upper = reward
    #                     if action_lower >= reward:
    #                         action_lower = reward
    #             uppers.append(action_upper)
    #             lowers.append(action_lower)

    #         self.uppers = uppers
    #         self.lowers = lowers

    def init_bounds(self):
        if self.uppers is None or self.lowers is None:
            lowers = []
            uppers = []
            for action in range(NUM_OF_ACTIONS):
                if action == LEFT or LEFT_FAST:
                    uppers.append(50)
                    lowers.append(-1000)
                elif action == STAY or STAY_FAST:
                    uppers.append(50)
                    lowers.append(-5)
                elif action == RIGHT or RIGHT_FAST:
                    uppers.append(50)
                    lowers.append(-1000)

            self.uppers = uppers
            self.lowers = lowers


    def get_average_reward(self, action):
        rewards = [scenario.get_reward(action) for scenario in self.scenarios]
        return 1.0/NUM_OF_SCENARIOS * np.sum(rewards) * GAMMA ** self.depth - LAMBDA

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

    def get_next_node(self, nodes_array):
        weu = None
        action = self.get_upper_action()
        node = None

        phi = len(self.scenarios)

        for k, v in self.children[action].items():
            child = nodes_array[v]
            child.init_bounds()
            upper, lower = child.upper_bound(), child.lower_bound()
            phi_prim = len(child.scenarios)
            excess = upper - lower - EPSILON * GAMMA**(-child.depth)
            weu_ = phi_prim / float(phi) * excess
            if weu is None:
                weu = weu_
                node_id = v
            else:
                if weu_ > weu:
                    weu = weu_
                    node_id = v
        return node_id, weu

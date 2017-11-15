import numpy as np
import pdb
import json
from copy import deepcopy

SEARCH_DEPTH = 10
GAMMA = 0.9
CARS = 5
UP = 0
STAY = 1
DOWN = 2
GOAL = 400

class Car(object):
    '''
    car moves horizontally, which means the lane is specified by te y coordinate
    consider the movement every 0.1 second
    '''
    def __init__(self, pos):
        self.x, self.y = pos
        self.bias = np.random.random()

    def move(self):
        self.vel = 2 if 0 <= self.y < 3 else 4
        new_x, new_y = self.x + self.vel, self.y
        self.x, self.y = new_x, new_y

    def ret_obs(self, rand_num):
        rand = (self.bias + rand_num) / 2
        if rand < 0.1:
            return [self.x - 2, self.y]
        elif 0.1 <= rand < 0.9:
            return [self.x, self.y]
        else:
            return [self.x + 2, self.y]

    def update_pos(self, new_pos):
        self.x, self.y = new_pos

    def switch_lane(self, action):
        '''
        action:
            'up', 'stay', 'down'
        '''
        if action == UP:
            self.y = self.y - 1
        elif action == DOWN:
            self.y = self.y + 1
        else:
            self.y = self.y
        
    def collide(self, car):
        if self.x == car.x and self.y == car.y:
            return True
        else:
            return False

    def pos2str(self):
        return str(self.x) + str(self.y)

    def set_pos(self, pos):
        self.x, self.y = pos
    

class Scenario(object):
    def __init__(self, rand_nums):
        self.rand_nums = rand_nums

    def init_states(self, cars, robot):
        self.cars = []
        for car in cars:
            # x, y = car.ret_obs(self.rand_nums[0])
            # self.cars.append(Car([x, y]))
            self.cars.append(deepcopy(car))
        self.robot = robot

    def check_collision(self):
        for car in self.cars:
            if self.robot.collide(car):
                return True
        return False

    def step(self, action):
        new_robot = deepcopy(self.robot)
        new_robot.switch_lane(action)
        new_robot.move()
        new_cars = []
        for car in self.cars:
            temp = deepcopy(car)
            temp.move()
            obs_x, obs_y = temp.ret_obs()
            temp.set_pos([obs_x, obs_y])
            new_cars.append(temp)

        new_rand_nums = self.rand_nums[1:]
        new_scenario = Scenario(new_rand_nums)
        new_scenario.init_states(new_cars, new_robot)
        collision = new_scenario.check_collision()
        goal = new_scenario.robot.x >= GOAL

        if collision:
            reward = -100
        elif action == DOWN: # encourage it to switch to the fast lane
            reward = 5
        elif goal:
            reward = 50
        else:
            reward = 0

        return goal, collision, reward, new_scenario

        
class Node(object):
    def __init__(self):
        self.scenarios = []
        self.children = []
        for _ in range(3):
            self.children.append(dict())

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def get_scenarios_len(self):
        return len(self.scenarios)

class DESPOT(object):
    def __init__(self, num_of_cars, belief):
        self.num_of_cars = num_of_cars
        self.belief = belief
        self.root = Node()

    def new_root_scenario(self, cars, robot, search_depth=SEARCH_DEPTH):
        self.search_depth = search_depth
        rand_nums = np.random.random(size=self.search_depth)
        scenario = Scenario(rand_nums)
        scenario.init_states(cars, robot)
        return scenario

    def search(self, root, scenario, depth):
        root.scenarios.append(scenario)
        if depth == self.search_depth:
            pass
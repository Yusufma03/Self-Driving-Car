import json
from discrete_action_pool import DiscreteActionPool
from discrete_observation_pool import DiscreteObservationPool
import numpy as np
from autocar_action import AutocarAction, ActionType
from autocar_state import AutocarState
from autocar_observation import AutocarObservation
from autocar_data import AutocarData
# from step_result import StepResult
from agent import Agent
import copy
import sys
import random



class AutoCarModel():
    def __init__(self):
        test = "test"

    def load_config(self):
        with open('pomcp_config.json', 'r') as fin:
            config = json.load(fin)

        self.discount = config['discount']
        self.n_epochs = config['n_epochs']
        self.max_steps = config['max_steps']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_minimum = config['epsilon_minimum']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_decay_step = config['epsilon_decay_step']
        self.n_sims = config['n_sims']
        self.timeout = config['timeout']
        self.ucb_coefficient = config['ucb_coefficient']
        self.n_start_states = config['n_start_states']
        self.min_particle_count = config['min_particle_count']
        self.max_particle_count = config['max_particle_count']
        self.max_depth = config['max_depth']
        self.action_selection_timeout = config['action_selection_timeout']
        # self.road_len = config['road_len']

        self.start_state = []

    def is_terminal(self, robot_pos):
        if robot_pos[0] < 5:
            return True
        else:
            return False

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)



    def get_all_actions(self):
        return [AutocarAction(ActionType.LEFT), AutocarAction(ActionType.STAY),
                AutocarAction(ActionType.RIGHT), AutocarAction(ActionType.FAST),
                AutocarAction(ActionType.SLOW)]


    def get_legal_actions(self, _):
        return self.get_all_actions()

    def reset_for_simulation(self):
        self.start_state = self.start_state

    def update(self, sim_data):
        pass


    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return AutocarData(self, AutocarObservation(self.start_state).obs)

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, state, action, robot_pos):
        if not isinstance(action, AutocarAction):
            action = AutocarAction(action)

        result = StepResult()

        current_robot_pos = copy.deepcopy(robot_pos)

        result.next_state, is_legal, robot_pos = self.make_next_state(state, action, robot_pos)
        result.action = action.copy()
        result.observation = self.make_observation(result.next_state)
        result.reward = self.make_reward(action, state, current_robot_pos)
        result.is_terminal = self.is_terminal(current_robot_pos)

        return result, is_legal, robot_pos


    def make_next_state(self, state, action, robot_pos):
        next_state = state

        is_legal = True

        if robot_pos[0] < 5:
            return state, is_legal, robot_pos

        if action.bin_number == ActionType.LEFT:
            if robot_pos[1] == 0:
                robot_pos[1] = 0
            else:
                robot_pos[1] -= 1
        elif action.bin_number == ActionType.RIGHT:
            if robot_pos[1] == 5:
                robot_pos[1] = 5
            else:
                robot_pos[1] += 1

        if robot_pos[1] < 3:
            robot_pos[0] -= 2
        else:
            if action.bin_number == ActionType.FAST:
                robot_pos[0] -= 4
            elif action.bin_number == ActionType.SLOW:
                robot_pos[0] -= 2
            else:
                robot_pos[0] -= 4

        for i in range(len(state)):
            for j in range(len(state[0])):
                for k in range(len(state[0][0])):
                    if k >=0 and k <= 2:
                        next_state[i][j - 2][k] = state[i][j][k]
                    else:
                        rand = random.uniform(0, 1)
                        if rand < 0.8:
                            next_state[i][j - 4][k] = state[i][j][k]
                        else:
                            next_state[i][j - 2][k] = state[i][j][k]



        return next_state, is_legal, robot_pos

    def make_reward(self, action, state, robot_pos):
        if robot_pos[0] < 5:
            return 50.0
        elif self.checkCollision(state, action, robot_pos) == 1:
            return -1000.0
        elif action.bin_number == ActionType.FAST and robot_pos[1] >=3 and robot_pos[1] <= 5:
            return 10.0
        elif action.bin_number == ActionType.RIGHT and robot_pos[1] <= 4:
            return 10.0
        else:
            return -0.2


    def checkCollision(self, state, action, robot_pos):
        # if robot_pos[1] == 3 and action.bin_number == ActionType.RIGHT:
        #     for i in range(len(state)):
        #         if state[i][robot_pos[0] + 4] == 1:
        #             return 1
        # elif robot_pos[1] == 5 and action.bin_number == ActionType.LEFT:
        #     for i in range(len(state)):
        #         if state[i][robot_pos[0] + 4] == 1:
        #             return 1
        # else:
        #     return 0

        for i in range(len(state)):
            for j in range(len(state[0])):
                for k in range(len(state[0][0])):
                    if action.bin_number == ActionType.RIGHT and robot_pos[1] < 5:
                        if robot_pos[1] >= 0 and robot_pos[1] < 1 and robot_pos[1]+1 == k and (robot_pos[0]+2) == j:
                            return 1
                        elif robot_pos[1] >= 2 and robot_pos[1] <= 4 and robot_pos[1]+1 == k and (robot_pos[0]+4) == j:
                            return 1
                        else:
                            return 0
                    if action.bin_number == ActionType.LEFT and robot_pos[1] > 0:
                        if robot_pos[1] >= 4 and robot_pos[1] <= 5 and robot_pos[1]-1 == k and (robot_pos[0]+4) == j:
                            return 1
                        elif robot_pos[1] > 0 and robot_pos[1] <= 3 and robot_pos[1]-1 == k and (robot_pos[0]+4) == j:
                            return 1
                        else:
                            return 0
                    if action.bin_number == ActionType.FAST and robot_pos[1] >= 3:
                        rand = np.random.uniform(0, 1)
                        if robot_pos[1] == k and (robot_pos[0]-2) == j and rand < 0.2:
                            return 1
                        else:
                            return 0
                    elif action.bin_number == ActionType.SLOW and robot_pos[1] <= 2:
                        rand = np.random.uniform(0, 1)
                        if robot_pos[1] == k and (robot_pos[0]+2) == j and rand < 0.2:
                            return 1
                        else:
                            return 0
                    return 0



    def make_observation(self, state):
        return AutocarObservation(state)

    def belief_update(self, old_belief, action, observation, robot_pos):
        # new_belief = [[0 for x in range(len(observation.obs[0]))] for y in range(len(observation.obs))]

        temp_belief = copy.deepcopy(old_belief)
        temp_belief2 = [[0 for x in range(len(observation.obs[0]))] for y in range(len(observation.obs))]


        for i in range(len(old_belief)):
            for j in range(len(old_belief[0])):
                if j > 4:
                    temp_belief2[i][j-4] = temp_belief[i][j]

        new_belief = np.array(old_belief) * np.array(old_belief)
        eta = np.sum(new_belief, axis=1)
        for i in range(len(eta)):
            new_belief[i] /= eta[i]
        return new_belief


    ##############################

    def generate_particles(self, previous_belief, action, obs, n_particles, prev_particles, robot_pos):
        particles = []
        action_node = previous_belief.action_map.get_action_node(action)
        if action_node is None:
            return particles
        else:
            obs_map = action_node.observation_map
        child_node = obs_map.get_belief(obs)

        print("n_particles")
        print(n_particles)

        while particles.__len__() < n_particles:
            print("n_particles")
            print(n_particles)
            state = random.choice(prev_particles)

            result, is_legal, robot_pos = self.generate_step(state, action, robot_pos)
            if obs_map.get_belief(result.observation) is child_node:
                particles.append(result.next_state)
        return particles

    def generate_particles_uninformed(self, previous_belief, action, obs, n_particles, robot_pos):

        particles = []
        obs_map = previous_belief.action_map.get_action_node(action).observation_map
        child_node = obs_map.get_belief(obs)

        while particles.__len__() < n_particles:
            state = self.sample_state_uninformed()
            result, is_legal, robot_pos = self.generate_step(state, action, robot_pos)
            if obs_map.get_belief(result.observation) is child_node:
                particles.append(result.next_state)
        return particles

class StepResult():
    def __init__(self):
        self.action = None
        self.observation = None
        self.reward = 0
        self.next_state = None
        self.is_terminal = 0



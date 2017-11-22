from history import Histories, HistoryEntry
from statistic import Statistic
from pomcp import POMCP
import time as systime
import copy
import sys

module = "agent"


class Agent:

    def __init__(self, model, solver):
        self.model = model
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)

    def run_pomcp(self, robot_pos):
        eps = self.model.epsilon_start  #eps = 0.5

        start_time_0 = systime.time()

        solver = POMCP(self)


        state = copy.deepcopy(self.model.start_state)

        reward = 0
        discounted_reward = 0
        discount = 0.9

        action_sequence = []

        robot_pos_sequence = []

        # print("start_state")
        # print(self.model.start_state[30][1])


        for i in range(self.model.max_steps):  #global max_steps = 200

            start_time = systime.time()

            curr_robot_pos = copy.deepcopy(robot_pos)

            # print("@@@@@@@@@@@@@@@")
            # print("self.robot_pos")
            # print(robot_pos)
            # print("curr_robot_pos")
            # print(curr_robot_pos)


            self.find_cars(state)

            # action will be of type Discrete Action
            action = solver.select_action(eps, start_time, curr_robot_pos)

            action_sequence.append(action.bin_number)

            # print("action in " + str(i) + " step")
            # print(action.bin_number)

            # print("#############")
            # print("state")
            # self.find_cars(state)
            # print("robot_pos")
            # print(robot_pos)


            # update epsilon
            if eps > self.model.epsilon_minimum:  #self.model.epsilon_minimum = 0.1
                eps *= self.model.epsilon_decay

            step_result, is_legal, curr_robot_pos = self.model.generate_step(state, action, robot_pos)

            ## robot_pos is correct


            # print("*$$$$$$$$$$$$$$$$$*")
            # print("self.robot_pos")
            # print(robot_pos)
            # print("result terminal")
            # print(step_result.is_terminal)
            # print("next state")
            # self.find_cars(step_result.next_state)
            # print("step_result.is_terminal")
            # print(step_result.is_terminal)
            # print("robot_pos")
            # print(robot_pos)
            # print("curr_robot_pos")
            # print(curr_robot_pos)
            # print("actions")
            # print(step_result.action.bin_number)
            # print(action.bin_number)
            # print("reward")
            # print(step_result.reward)


            # if i > 3:
            #     sys.exit()


            reward += step_result.reward
            discounted_reward += discount * step_result.reward

            discount *= self.model.discount
            state = step_result.next_state


            if not step_result.is_terminal or not is_legal:
                solver.update(step_result, robot_pos)

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                              step_result.action, step_result.observation, step_result.next_state)

            # if step_result.is_terminal or not is_legal:
            if step_result.is_terminal:
                print("program ending with terminal state")
                break

        self.results.time.mean += (systime.time() - start_time_0)
        self.results.update_reward_results(reward, discounted_reward)


        self.experiment_results.time.running_total += self.results.time.running_total
        self.experiment_results.undiscounted_return.count += self.results.undiscounted_return.count - 1
        self.experiment_results.undiscounted_return.running_total += self.results.undiscounted_return.running_total
        self.experiment_results.discounted_return.count += self.results.discounted_return.count - 1
        self.experiment_results.discounted_return.running_total += self.results.discounted_return.running_total

        print("reward")
        print(reward)
        print("self.results.time.mean")
        print(self.results.time.mean)
        return action_sequence

    def find_cars(self, state):
        cars_pos = []
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == 1:
                    cars_pos.append([i,j])
        # print("cars_pos")
        # print(cars_pos)

class Results():

    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.running_total += r
        self.discounted_return.running_total += dr

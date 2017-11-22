import time
import numpy as np
from action_selection import ucb_action
from belief_tree_solver import BeliefTreeSolver
import sys

class POMCP(BeliefTreeSolver):

    UCB_N = 1000
    UCB_n = 100

    def __init__(self, agent):
        super(POMCP, self).__init__(agent)

        self.fast_UCB = [[None for _ in range(POMCP.UCB_n)] for _ in range(POMCP.UCB_N)]

        for N in range(POMCP.UCB_N):
            for n in range(POMCP.UCB_n):
                if n is 0:
                    self.fast_UCB[N][n] = np.inf
                else:
                    self.fast_UCB[N][n] = agent.model.ucb_coefficient * np.sqrt((np.log(N + 1) / n))


    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        assert self.fast_UCB is not None
        if total_visit_count < POMCP.UCB_N and action_map_entry_visit_count < POMCP.UCB_n:
            return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

        if action_map_entry_visit_count == 0:
            return np.inf
        else:
            return self.model.ucb_coefficient * np.sqrt((log_n / action_map_entry_visit_count))

    def select_action(self, eps, start_time, robot_pos):
        if self.disable_tree:
            # print("rollout_search")
            self.rollout_search(self.belief_tree_index, robot_pos)
        else:
            # print("monte_carlo_approx")
            self.monte_carlo_approx(eps, start_time, robot_pos)
        return ucb_action(self, self.belief_tree_index, True)

    def simulate(self, belief_node, eps, start_time, robot_pos):
        return self.traverse(belief_node, 0, start_time, robot_pos)

    def traverse(self, belief_node, tree_depth, start_time, robot_pos):

        delayed_reward = 0

        state = belief_node.sample_particle()

        if time.time() - start_time > self.model.action_selection_timeout:
            print("Time expired in traverse")
            return 0

        action = ucb_action(self, belief_node, False)


        if tree_depth >= self.model.max_depth:  # self.model.max_depth == 100
            return 0

        # print("robot_pos 1")
        # print(robot_pos)

        step_result, is_legal, robot_pos = self.model.generate_step(state, action, robot_pos)

        # no change of action

        # print("robot_pos 2")
        # print(robot_pos)


        # print("action")
        # print(action.bin_number)
        # print("robot_pos")
        # print(robot_pos)
        # print("obser")
        # print(step_result.observation.obs)

        child_belief_node = belief_node.child(action, step_result.observation, robot_pos)
        if child_belief_node is None and not step_result.is_terminal and belief_node.action_map.total_visit_count > 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation, robot_pos)

        if not step_result.is_terminal or not is_legal:
            print("ever come here? not is terminal?")
            tree_depth += 1
            if child_belief_node is not None:
                print("traverse")
                # Add S' to the new belief node
                # Add a state particle with the new state
                if child_belief_node.state_particles.__len__() < self.model.max_particle_count:
                    child_belief_node.state_particles.append(step_result.next_state)
                delayed_reward = self.traverse(child_belief_node, tree_depth, start_time, robot_pos)
            else:
                print("rollout")
                delayed_reward = self.rollout(belief_node, robot_pos)
            tree_depth -= 1
        # else:
            # print("((( we have reached terminal state)))")

        action_mapping_entry = belief_node.action_map.get_entry(action.bin_number)

        q_value = action_mapping_entry.mean_q_value

        q_value += (step_result.reward + (self.model.discount * delayed_reward) - q_value)

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        # print("q_value")
        # print(q_value)

        return q_value
from builtins import range
import time
import random
from belief_tree import BeliefTree

module = "BeliefTreeSolver"


class BeliefTreeSolver():
    def __init__(self, agent):
        self.model = agent.model
        self.history = agent.histories.create_sequence()
        self.disable_tree = False

        self.belief_tree = BeliefTree(agent)

        self.belief_tree.reset()
        self.belief_tree.initialize()

        particle = agent.model.start_state
        self.belief_tree.root.state_particles.append(particle)
        self.belief_tree_index = self.belief_tree.root.copy()

    def monte_carlo_approx(self, eps, start_time, robot_pos):

        for i in range(self.model.n_sims):  # self.model.n_sims = 500
            self.model.reset_for_simulation()
            self.simulate(self.belief_tree_index, eps, start_time, robot_pos)

    def prune(self, belief_node):
        start_time = time.time()
        self.belief_tree.prune_siblings(belief_node)
        elapsed = time.time() - start_time

    def rollout_search(self, belief_node, robot_pos):
        legal_actions = belief_node.data.generate_legal_actions()

        for i in range(legal_actions.__len__()):
            state = belief_node.sample_particle()
            action = legal_actions[i % legal_actions.__len__()]

            step_result, is_legal, robot_pos  = self.model.generate_step(state, action, robot_pos)

            if not step_result.is_terminal:
                child_node, added = belief_node.create_or_get_child(step_result.action, step_result.observation, robot_pos)
                child_node.state_particles.append(step_result.next_state)
                delayed_reward = self.rollout(child_node, robot_pos)
            else:
                delayed_reward = 0

            action_mapping_entry = belief_node.action_map.get_entry(step_result.action.bin_number)

            q_value = action_mapping_entry.mean_q_value

            q_value += (step_result.reward + self.model.discount * delayed_reward - q_value)

            action_mapping_entry.update_visit_count(1)
            action_mapping_entry.update_q_value(q_value)

    def rollout(self, belief_node, robot_pos):
        legal_actions = belief_node.data.generate_legal_actions()

        if not isinstance(legal_actions, list):
            legal_actions = list(legal_actions)

        state = belief_node.sample_particle()
        is_terminal = False
        discounted_reward_sum = 0.0
        discount = 1.0
        num_steps = 0

        while num_steps < self.model.max_depth and not is_terminal:
            legal_action = random.choice(legal_actions)
            step_result, is_legal, robot_pos = self.model.generate_step(state, legal_action, robot_pos)
            is_terminal = step_result.is_terminal
            discounted_reward_sum += step_result.reward * discount
            discount *= self.model.discount
            state = step_result.next_state
            legal_actions = self.model.get_legal_actions(state)
            num_steps += 1

        return discounted_reward_sum

    def update(self, step_result, robot_pos, prune=True):
        self.model.update(step_result)

        child_belief_node = self.belief_tree_index.get_child(step_result.action, step_result.observation)

        if child_belief_node is None:
            action_node = self.belief_tree_index.action_map.get_action_node(step_result.action)
            if action_node is None:
                self.disable_tree = True
                return

            obs_mapping_entries = list(action_node.observation_map.child_map.values())

            for entry in obs_mapping_entries:
                if entry.child_node is not None:
                    child_belief_node = entry.child_node
                    break

        if child_belief_node.state_particles.__len__() < self.model.max_particle_count:

            num_to_add = self.model.max_particle_count - child_belief_node.state_particles.__len__()


            child_belief_node.state_particles += self.model.generate_particles(self.belief_tree_index, step_result.action,
                                                                               step_result.observation, num_to_add,
                                                                               self.belief_tree_index.state_particles, robot_pos)


            if child_belief_node.state_particles.__len__() == 0:
                child_belief_node.state_particles += self.model.generate_particles_uninformed(self.belief_tree_index,
                                                                                              step_result.action,
                                                                                              step_result.observation,
                                                                                        self.model.min_particle_count, robot_pos)

        if child_belief_node is None or child_belief_node.state_particles.__len__() == 0:
            self.disable_tree = True
            return

        self.belief_tree_index = child_belief_node
        if prune:
            self.prune(self.belief_tree_index)
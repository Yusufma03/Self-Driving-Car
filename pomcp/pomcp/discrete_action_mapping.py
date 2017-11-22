from action_node import ActionNode
import numpy as np


class DiscreteActionMapping():
    def __init__(self, belief_node_owner, discrete_action_pool, bin_sequence):
        self.owner = belief_node_owner
        self.pool = discrete_action_pool
        self.number_of_bins = self.pool.all_actions.__len__()
        self.entries = {}
        self.bin_sequence = bin_sequence
        self.number_of_children = 0
        self.total_visit_count = 0

        for i in range(0,self.number_of_bins):
            entry = DiscreteActionMappingEntry()
            entry.bin_number = i
            entry.map = self
            entry.is_legal = False
            self.entries.__setitem__(i, entry)

        for bin_number in self.bin_sequence:
            self.entries.get(bin_number).is_legal = True

    def copy(self):
        action_map_copy = DiscreteActionMapping(self.owner, self.pool, list(self.bin_sequence))
        action_map_copy.number_of_children = self.number_of_bins
        action_map_copy.entries = self.entries.copy()
        action_map_copy.number_of_children = self.number_of_children
        action_map_copy.total_visit_count = self.total_visit_count
        return action_map_copy

    def get_action_node(self, action):
        return self.entries.get(action.bin_number).child_node

    def create_action_node(self, action):
        entry = self.entries.get(action.bin_number)
        entry.child_node = ActionNode(entry)
        self.number_of_children += 1
        return entry.child_node

    def get_child_entries(self):
        return_entries = []
        for i in range(0, self.number_of_bins):
            entry = self.entries.get(i)
            if entry.child_node is not None:
                return_entries.append(entry)
        return return_entries


    def get_entry(self, action_bin_number):
        return self.entries.get(action_bin_number)

    def update(self):
        self.bin_sequence = self.pool.create_bin_sequence(self.owner)

        for entry in list(self.entries.values()):
            entry.is_legal = False

        for bin_number in self.bin_sequence:
            self.entries.get(bin_number).is_legal = True


class DiscreteActionMappingEntry():
    def __init__(self):
        self.bin_number = -1
        self.map = None
        self.child_node = None
        self.visit_count = 0
        self.total_q_value = 0
        self.mean_q_value = 0
        self.is_legal = False
        self.preferred_action = False

    def get_action(self):
        return self.map.pool.sample_an_action(self.bin_number)

    def update_visit_count(self, delta_n_visits):
        if delta_n_visits == 0:
            return

        self.visit_count += delta_n_visits
        self.map.total_visit_count += delta_n_visits

        return self.visit_count

    def update_q_value(self, delta_total_q, delta_n_visits=0):
        if delta_total_q == 0:
            return False

        assert np.isfinite(delta_total_q)

        if delta_n_visits != 0:
            self.update_visit_count(delta_n_visits)

        if self.preferred_action and delta_total_q < 0:
            delta_total_q = -delta_total_q

        self.total_q_value += delta_total_q

        old_mean_q = self.mean_q_value

        self.mean_q_value = self.total_q_value / self.visit_count

        return self.mean_q_value != old_mean_q







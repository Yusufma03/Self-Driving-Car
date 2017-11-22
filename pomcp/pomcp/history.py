class HistoryEntry():

    def __init__(self, owning_sequence, id):
        self.owning_sequence = owning_sequence
        self.associated_belief_node = None
        self.id = id
        self.state = None
        self.action = None
        self.observation = None
        self.reward = 0

    def register_node(self, node):
        if self.associated_belief_node is node:
            return

        if self.associated_belief_node is not None:
            self.associated_belief_node.remove_particle(self)
            self.associated_belief_node = None
        if node is not None:
            self.associated_belief_node = node
            self.associated_belief_node.add_particle(self)

    def register_state(self, state):
        if self.state is state:
            return

        if self.state is not None:
            self.state = None
        if state is not None:
            self.state = state

    @staticmethod
    def register_entry(current_entry, node, state):
        current_entry.register_state(state)
        current_entry.register_node(node)

    @staticmethod
    def update_history_entry(h, r, a, o, s):
        h.reward = r
        h.action = a
        h.observation = o
        h.register_entry(h, None, s)


class HistorySequence():

    def __init__(self, id):
        self.id = id
        self.entry_sequence = []

    def get_states(self):
        states = []
        for i in self.entry_sequence:
            states.append(i.state)
        return states

    def add_entry(self):
        new_entry = HistoryEntry(self, self.entry_sequence.__len__())
        self.entry_sequence.append(new_entry)
        return new_entry




class Histories():

    def __init__(self):
        self.sequences_by_id = []

    def create_sequence(self):
        hist_seq = HistorySequence(self.sequences_by_id.__len__())
        self.sequences_by_id.append(hist_seq)
        return hist_seq


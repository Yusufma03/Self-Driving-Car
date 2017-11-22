from discrete_action_mapping import DiscreteActionMapping


class DiscreteActionPool():
    def __init__(self, model):
        self.all_actions = model.get_all_actions()

    def create_action_mapping(self, belief_node):
        return DiscreteActionMapping(belief_node, self, self.create_bin_sequence(belief_node))

    def sample_an_action(self, bin_number):
        return self.all_actions[bin_number]

    @staticmethod
    def create_bin_sequence(belief_node):
        return belief_node.data.legal_actions()

from belief_node import BeliefNode


class BeliefTree():
    def __init__(self, agent):
        self.agent = agent
        self.root = None

    def reset(self):
        self.prune_tree(self)
        self.root = BeliefNode(self.agent, None, None)
        return self.root

    def reset_root_data(self):
        self.root.data = self.agent.model.create_root_historical_data(self.agent)


    def initialize(self, init_value=None):
        self.reset_root_data()
        self.root.action_map = self.agent.action_pool.create_action_mapping(self.root)

    def prune_tree(self, bt):
        self.prune_node(bt.root)
        bt.root = None

    def prune_node(self, bn):
        if bn is None:
            return

        bn.parent_entry = None

        bn.action_map.owner = None

        action_mapping_entries = bn.action_map.get_child_entries()

        for entry in action_mapping_entries:
            entry.child_node.parent_entry = None
            entry.map = None
            entry.child_node.observation_map.owner = None
            for observation_entry in list(entry.child_node.observation_map.child_map.values()):
                self.prune_node(observation_entry.child_node)
                observation_entry.map = None
                observation_entry.child_node = None
            entry.child_node.observation_map = None
            entry.child_node = None
        bn.action_map = None

    def prune_siblings(self, bn):
        if bn is None:
            return

        parent_belief = bn.get_parent_belief()

        if parent_belief is not None:

            for action_mapping_entry in parent_belief.action_map.get_child_entries():
                for obs_mapping_entry in action_mapping_entry.child_node.observation_map.get_child_entries():

                    if obs_mapping_entry.child_node is not bn:
                        self.prune_node(obs_mapping_entry.child_node)
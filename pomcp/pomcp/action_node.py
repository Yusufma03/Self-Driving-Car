class ActionNode(object):

    def __init__(self, parent_entry=None):
        if parent_entry is not None:
            self.parent_entry = parent_entry
        else:
            self.parent_entry = None
        self.observation_map = None


    ##############

    def set_mapping(self, obs_mapping):
        self.observation_map = obs_mapping

    ##############


    def create_or_get_child(self, obs):
        child_node = self.observation_map.get_belief(obs)
        added = False
        if child_node is None:
            child_node = self.observation_map.create_belief(obs)
            added = True
        return child_node, added

    def get_parent_belief(self):
        return self.parent_entry.get_mapping().get_owner()

    def get_child(self, obs):
        return self.observation_map.get_belief(obs)





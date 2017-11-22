from discrete_observation_mapping import DiscreteObservationMap


class DiscreteObservationPool():

    def __init__(self, agent):
        self.agent = agent

    def create_observation_mapping(self, action_node):
        return DiscreteObservationMap(action_node, self.agent)

import random

class BeliefNode(object):
    def __init__(self, solver, id=None, parent_entry=None):
        if id is None:
            self.id = -1
        else:
            self.id = id

        self.solver = solver
        self.data = None
        self.depth = -1
        self.action_map = None
        self.state_particles = []

        if parent_entry is not None:
            self.parent_entry = parent_entry
            self.depth = self.get_parent_belief().depth + 1
        else:
            self.parent_entry = None
            self.depth = 0

    def copy(self):
        bn = BeliefNode(self.solver, self.id, self.parent_entry)
        bn.data = self.data.copy()
        bn.action_map = self.action_map
        bn.state_particles = self.state_particles
        return bn


    def sample_particle(self):
        return random.choice(self.state_particles)


    def get_parent_belief(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner.parent_entry.map.owner
        else:
            return None



    def get_child(self, action, obs):
        node = self.action_map.get_action_node(action)
        if node is not None:
            return node.get_child(obs)
        else:
            return None

    def child(self, action, obs, robot_pos):
        node = self.action_map.get_action_node(action)
        if node is not None:
            child_node = node.get_child(obs)
            if child_node is None:
                return None
            child_node.data.update(child_node.get_parent_belief(), robot_pos)
            return child_node
        else:
            return None


    def create_or_get_child(self, action, obs, robot_pos):
        action_node = self.action_map.get_action_node(action)
        if action_node is None:
            action_node = self.action_map.create_action_node(action)
            action_node.set_mapping(self.solver.observation_pool.create_observation_mapping(action_node))
        child_node, added = action_node.create_or_get_child(obs)

        if added:
            if self.data is not None:
                child_node.data = self.data.create_child(action, obs, robot_pos)
            child_node.action_map = self.solver.action_pool.create_action_mapping(child_node)
        else:
            child_node.data.update(child_node.get_parent_belief(), robot_pos)
        return child_node, added

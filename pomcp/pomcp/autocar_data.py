from autocar_action import ActionType
import numpy as np


class AutocarData():

    def __init__(self, model, belief_dis):
        self.model = model
        self.belief_dis = belief_dis
        self.legal_actions = self.generate_legal_actions
        # self.robot_pos = robot_pos

    ##################
    def create_child(self, action, observation, robot_pos):
        next_data = self.copy()
        next_data.belief_dis = self.model.belief_update(self.belief_dis, action, observation, robot_pos)
        return next_data


    def generate_legal_actions(self):
        return [ActionType.LEFT, ActionType.STAY, ActionType.RIGHT, ActionType.FAST, ActionType.SLOW]

    def copy(self):
        dat = AutocarData(self.model, self.belief_dis)
        return dat

    def update(self, other_belief):
        self.door_probabilities = other_belief.data.door_probabilities



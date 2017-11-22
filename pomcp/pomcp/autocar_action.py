class ActionType():
    LEFT = 0
    STAY = 1
    RIGHT = 2
    FAST = 3
    SLOW = 4


class AutocarAction():
    def __init__(self, action_type):
        self.bin_number = action_type

    def copy(self):
        return AutocarAction(self.bin_number)

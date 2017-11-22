class AutocarState():
    def __init__(self, curr_state):
        self.state = curr_state

    def copy(self):
        return AutocarState(self.state)
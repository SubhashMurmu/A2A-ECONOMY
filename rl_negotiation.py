import random

class RLNegotiator:
    def __init__(self):
        self.q_table = {}

    def get_action(self, state):
        return self.q_table.get(state, random.choice(["accept", "reject", "counter"]))
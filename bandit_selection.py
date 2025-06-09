import random

class BanditSelector:
    def __init__(self, agents):
        self.agents = agents
        self.rewards = {agent: 1 for agent in agents}

    def select(self):
        return max(self.agents, key=lambda a: self.rewards[a] + random.random())

    def update(self, agent, reward):
        self.rewards[agent] += reward
        
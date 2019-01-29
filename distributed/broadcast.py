from modelConfig import params

class Broadcast:
    def __init__(self, goals, agent, optionID):
        self.goals = goals
        self.agent = agent
        self.optionID = optionID

    def broadcastBasedOnQ(self, Q0, Q1):
        """An agent broadcasts if the agent is at any goal or the intra-option value for
        no broadcast (Q0) is less than that with broadcast (Q1)"""

        return (self.agent.state in self.goals) or (Q0 < Q1)


    def randomBroadcast(self, state):
        n = params['train']['seed'].uniform(0,1)
        if n < 0.1:
            return 1  # broadcast with probability 0.1
        return 0
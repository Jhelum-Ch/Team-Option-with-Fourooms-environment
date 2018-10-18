from option import Option

# Agent for the FourRooms environment

class Agent:
    def __init__(self):
        super(Agent, self).__init__()
        # name
        self.name = ''
        # state
        self.state = None
        # Current running option o^j (index corresponding to option in option space O)
        self.option = None

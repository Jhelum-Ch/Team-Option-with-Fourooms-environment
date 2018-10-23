from option import Option

# Agent for the FourRooms environment

class Agent:
    def __init__(self, ID, name = ""):
        # name
        self.name = name
        # ID (required)
        self.ID = ID
        # state
        self.state = None
        # Current running option o^j (index corresponding to option in option space O)
        self.option = None

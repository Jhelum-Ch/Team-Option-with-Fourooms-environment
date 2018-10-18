import numpy as np

# Option of an agent: tuple(action-policy, termination function, broadcast? )
# Initiation set omitted as it does not seem to be used.

class Option(object):
    def __init__(self):
        # initiation set
        # self.initiation_set = None
        # action-policy pi
        self.action = None
        # termination function beta
        self.termination = None
        # broadcast action
        self.broadcast = 0

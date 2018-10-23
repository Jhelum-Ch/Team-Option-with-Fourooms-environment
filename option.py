import numpy as np

# Option of an agent: tuple(action-policy, termination function, broadcast? )
# Initiation set omitted as it does not seem to be used.

class Option(object):
    def __init__(self):
        # initiation set
        self.initiation = None
        # action-policy pi
        self.policy = None
        # termination function beta
        self.termination = None
        # broadcast action (0: does not broadcast. 1: broadcasts.)
        self.broadcast = 0


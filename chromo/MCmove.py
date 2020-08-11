"""
MCmove class

Creates a Monte Carlo move object that contains parameters for the particular move
"""

import numpy as np


class MCmove:
    type = "Simulation conditions for the MC moves"

    def __init__(self, mcmove_count):
        self.name = "MC move type " + str(mcmove_count)
        self.mcmove_on = True
        self.amp_move = 2 * np.pi
        self.num_per_cycle = 1
        self.amp_bead = 10
        self.num_attempt = 0
        self.num_success = 0
        self.move_on = True

    def __str__(self):
        return f"{self.name} is a Monte Carlo move"



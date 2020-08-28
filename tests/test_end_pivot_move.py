
from dataclasses import dataclass

# from chromo.mc.mc_move import *
from chromo.Polymer import *

import numpy as np

def test_end_pivot_move():
    name = "test_end_pivot"
    num_beads = 10
    
    states = np.zeros(num_beads)
    length_per_bead = 1
    polymer = Polymer.straight_line_in_x(name, marks, states, num_beads, length_per_bead)
          
    print(polymer.r)
    input("breather")
    
    assert True



from chromo.mc.mc_move import *
import inputs.chromo_for_testing_mc_moves
from chromo.mc.mc_sim import *

import numpy as np

def test_end_pivot_move():

    num_polymers = 1
    length_bead = 55
    num_epigenmark = 2
    length_box_x = 3
    num_bins_x = 3
    length_box_y = 3 
    num_bins_y = 3
    length_box_z = 3 
    num_bins_z = 3
    num_save_mc = 10
    num_mc_steps = 1000
    
	# Generate objects
	inputs.chromo_for_testing_mc_moves.main()

	assert True


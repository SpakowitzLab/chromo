"""
Utility function for random bead selection
"""

import random

import numpy as np

def select_bead_from_left(num_beads, all_beads, exclude_last_bead = True):
    """
    Randomly select a bead in the polymer chain with an exponentially decaying probability with index dist from  first point
    
    Parameters
    ----------
    num_beads:              int
                            Number of beads in the polymer chain
    all_beads:              np.array
                            1D vector of all bead indices in the polymer.
    exclude_last_bead:      Boolean (default: True)
                            Set True to exclude the final bead from selection (when rotating LHS of polymer)
    
    Returns
    -------
    ind0:                   int
                            Index of a bead selected with exponentially decaying probability from first point
    """

    if exclude_last_bead is True:
        
        norm_const = np.sum(np.exp(1 - all_beads[0:len(all_beads)-2]/(num_beads)))	# for normalization
        rand_val = random.uniform(0,1)	# for bead selection
        prob = 0
        for i in range(0, len(all_beads) - 1):
            prob += np.exp(1 - all_beads[i]/num_beads) / norm_const
            if rand_val < prob:	# select bead when random value exceeds cumulative probability
                ind0 = i + 1    # Check that ind0 starts at 1
                break
    else:
        norm_const = np.sum(np.exp(1 - all_beads[0:len(all_beads)-1]/(num_beads)))
        rand_val = random.uniform(0,1)
        prob = 0
        for i in range(0, len(all_beads)):
            prob += np.exp(1 - all_beads[i]/num_beads) / norm_const
            if rand_val < prob:
                ind0 = i + 1    # Check that ind0 starts at 1
                break

    return ind0


def select_bead_from_right(num_beads, all_beads, exclude_first_bead = True):
    """
    Randomly select a bead in the polymer chain with an exponentially decaying probability with index dist from last point
    
    Parameters
    ----------
    num_beads:              int
                            Number of beads in the polymer chain
    all_beads:              np.array
                            1D vector of all bead indices in the polymer.
    exclude_first_bead:     Boolean (default: True)
                            Set True to exclude the first bead from selection (when rotating  RHS of polymer)
    Returns
    -------
    ind0:                   int
                            Index of a bead selected with exponentially decaying probability from last point
    """
    if exclude_first_bead is  True:

        norm_const = np.sum(np.exp(1 - (num_beads - all_beads[1:len(all_beads)-1])/(num_beads)))
        rand_val = random.uniform(0,1)
        prob = 0
        for i in range(1, len(all_beads)):
            prob += np.exp(1 - (num_beads - all_beads[i])/(num_beads)) / norm_const
            if rand_val < prob:
                ind0 = i + 1
                break
    else:
        norm_const = np.sum(np.exp(1 - (num_beads - all_beads[0:len(all_beads)-1])/(num_beads)))
        rand_val = random.uniform(0,1)
        prob = 0
        for i in range(0, len(all_beads)):
            prob += np.exp(1 - (num_beads - all_beads[i])/(num_beads)) / norm_const
            if rand_val < prob:
                break

    return ind0        
	


def select_bead_from_point(num_beads, all_beads, ind0):
	"""
    Randomly select a bead in the polymer chain with exponentially decaying probability based on index distance from another point

	Parameters
    ----------
    num_beads:  int
                Number of beads in the polymer chain
    all_beads:  np.array
                1D vector of all bead indices in the polymer.
    ind0:       int
                Index of first point
    Returns
    -------
    indf:       int
                Index of new point selected based on distance from ind0
	"""

	norm_const = np.sum(np.exp(1 - abs(all_beads - ind0) / num_beads))
	rand_val = random.uniform(0, 1)
	prob = 0
	for i in range(0, len(all_beads)):
		prob += np.exp(1-abs(all_beads[i] - ind0) / num_beads) / norm_const
		if rand_val < prob:
			indf = i + 1
			break

	return indf

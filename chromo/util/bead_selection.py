"""
Utility function for random bead selection
"""

import random

import numpy as np

def exponential_random_int(window):
    """ 
    Randomly select an exponentially distributed integer
    Most likely to return values close to zero

    Parameters
    ----------
    window:         int
                    Bead window size for selection

    Returns
    -------
    ind:            int
                    Exponentially distributed integer from zero
    """
    
    accept_ind = False
    while accept_ind == False:
        unif_rand = random.uniform(0, 1)    # Generate random number of obtaining index
        ind = int(-1.0 * np.log(-unif_rand + 1) * window)
        ind = abs(ind)
        if ind < window:
            accept_ind = True

    return(ind)


def select_bead_from_left(window, all_beads, exclude_last_bead = True):
    """
    Randomly select index exponentially decaying from left

    Parameters
    ----------
    window:                 int
                            Bead window size for selection
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
        is_last_bead = True
        while is_last_bead == True:
            ind0 = exponential_random_int(window)
            if ind0 != all_beads[-1]:
                is_last_bead = False
    else:
        ind0 = exponential_random_int(window)

    return ind0


def select_bead_from_right(window, all_beads, exclude_first_bead = True):
    """
    Randomly select index exponentially decaying from right

    Parameters
    ----------
    window:                 int
                            Bead window size for selection
    all_beads:              np.array
                            1D vector of all bead indices in the polymer.
    exclude_first_bead:     Boolean (default: True)
                            Set True to exclude the first bead from selection (when rotating  RHS of polymer)
    Returns
    -------
    ind0:                   int
                            Index of a bead selected with exponentially decaying probability from last point
    """
    
    if exclude_first_bead is True:
        is_first_bead = True
        while is_first_bead == True:
            ind0 = np.max(all_beads) - exponential_random_int(window)
            if ind0 != all_beads[0]:
                is_first_bead = False
    else:
        ind0 = np.max(all_beads) - exponential_random_int(window)

    return ind0


def select_bead_from_point(window, all_beads, ind0):
    """
    Randomly select index ewponentially decaying from point ind0

    Parameters
    ----------
    window:     int
                Bead window size for selection  
    all_beads:  np.array
                1D vector of all bead indices in the polymer.
    ind0:       int
                Index of first point
    Returns
    -------
    indf:       int
                Index of new point selected based on distance from ind0
    """

    side = random.randint(0, 1)     # randomly select a side of the polymer to select from
    window_side = round(window / 2)

    if side == 0:       # LHS
        all_beads = all_beads[0:window_side] 
        indf = select_bead_from_right(window_side, all_beads, exclude_first_bead = False)
    else:               # RHS
        all_beads = all_beads[0:window_side] 
        indf = select_bead_from_left(window_side, all_beads, exclude_last_bead = False) + ind0

    return(indf)


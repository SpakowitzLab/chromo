"""
Utility functions for random bead selection.
"""

import random
import sys

import numpy as np


def capped_exponential(window, cap=np.inf):
    """
    Select exponentially distributed integer below some capped value.

    Parameters
    ----------
    window : int
        Mean of the exponential distribution being sampled
    cap : int
        Maximum value of exponentially sampled integer 
    Returns
    -------
    r : int
        Exponentially sampled random integer less than capped value
    """

    r = np.random.geometric(1/window)
    while r > cap:
        r = np.random.geometric(1/window)
    return r

def select_bead_from_left(window, N_beads, exclude_last_bead=True):
    """
    Randomly select index exponentially decaying from left.

    Parameters
    ----------
    window : int
        Bead window size for selection
    N_beads : int
        Number of beads in the polymer chain
    exclude_last_bead : boolean
        Set to True to exclude final bead from selection, such as when rotating LHS of polymer (default = True) 
    Returns
    -------
    int
        Index of a bead selected with exponentially decaying probability from first point
    """
    if exclude_last_bead == True:
        if window == N_beads:
            window -= 1
        N_beads -= 1
        
    if window > N_beads:
        raise ValueError("Bead selection window size must be less than polymer length")
    
    return capped_exponential(window, N_beads)


def select_bead_from_right(window, N_beads, exclude_first_bead = True):
    """
    Randomly select index exponentially decaying from right.

    Parameters
    ----------
    window : int
        Bead window size for selection
    N_beads : int
        Number of beads in the polymer chain
    exclude_first_bead : boolean
        Set to True to exclude first bead from selection, such as when rotating RHS of polymer (default = True) 
    Returns
    -------
    int
        Index of a bead selected with exponentially decaying probability from last point
    """
   
    dist_from_RHS = select_bead_from_left(window, N_beads, exclude_first_bead)
    return N_beads - dist_from_RHS


def select_bead_from_point(window, N_beads, ind0):
    """
    Randomly select index exponentially decaying from point ind0.

    Parameters
    ----------
    window : int
        Bead window size for selection  
    N_beads : int
        Number of beads in the polymer chain
    ind0 : int
        Index of first point
    Returns
    -------
    int
        Index of new point selected based on distance from ind0
    """

    if window > N_beads:
        raise ValueError("Bead selection window size must be less than polymer length.")

    side = random.randint(0, 1)     # randomly select a side of the polymer to select from
    window_side = round(window/2)
    if window_side == 0 : return ind0   # Do not perform move if move window size is zero

    if side == 0:       # LHS
        return select_bead_from_right(window_side, ind0, exclude_first_bead = False)
    else:               # RHS
        return select_bead_from_left(window_side, N_beads - ind0, exclude_last_bead = False) + ind0





"""Utility functions for random bead selection.
"""

import pyximport
pyximport.install()

# Built-in Modules
import sys
import random
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log10

# External Modules
import numpy as np


cdef long capped_exponential(long window, long cap):
    """Select exponentially distributed integer below some capped value.

    Specify the geometric distribution such that 99% of outcomes fall below
    window. Do this by manipulating the argument of `np.random.geometric()`
    and selecting a success probability such that the CDF of the geometric
    distribution at `window` equals 0.99. This will significantly improve
    runtime, particularly at the while loop.

    Notes
    -----
    The function can also be written using numpy's genometric random sampler
    using the following code. The code is slightly slower than directly coding
    the exponential random sampler in cython.

    # Exponential random sampler using numpy's random geometric method
    cdef double eCDF_val_at_window, p
    cdef long r

    eCDF_val_at_window = 0.999
    p = 1 - (1 - eCDF_val_at_window) ** (1 / (<double>window + 1))
    r = np.random.geometric(p)
    if window < 1:
        raise ValueError("Selection window ", window, " is less than 1.")

    Parameters
    ----------
    window : int
        Width of the exponential distribution being sampled
    cap : int
        Maximum value of exponentially sampled integer

    Returns
    -------
    r : int
        Exponentially sampled random integer less than capped value
    """
    cdef long r

    r = <long>(-log10(
        (<double>rand() / RAND_MAX) + 0.00001
    ) * window * 0.45 + 1.0001)

    while r > cap:
        r = <long>(-log10(
            (<double>rand() / RAND_MAX) + 0.00001
        ) * window * 0.45 + 1.0001)

    return r

cpdef long from_left(long window, long N_beads):
    """Randomly select index exponentially decaying from left.

    Parameters
    ----------
    window : int
        Bead window size for selection (must be less than polymer length
        `N_beads`)
    N_beads : int
        Number of beads in the polymer chain

    Returns
    -------
    int
        Index of a bead selected with exponentially decaying probability from
        first position in the chain.
    """
    if window > N_beads:
        raise ValueError(
            "Bead selection window size must be less than polymer length"
        )
    return capped_exponential(window, window)


cpdef long from_right(long window, long N_beads):
    """Randomly select index exponentially decaying from right.

    Parameters
    ----------
    window : int
        Bead window size for selection (must be less than polymer length
        `N_beads`)
    N_beads : int
        Number of beads in the polymer chain

    Returns
    -------
    int
        Index of a bead selected with exponentially decaying probability from
        last position in the chain.
    """

    cdef long dist_from_RHS = from_left(window, N_beads)
    return N_beads - dist_from_RHS


cpdef long from_point(long window, long N_beads, long ind0):
    """Randomly select index w/ exponentially decaying probability from point.

    Parameters
    ----------
    window : int
        Bead window size for selection (must be less than polymer length
        `N_beads`)
    N_beads : int
        Number of beads in the polymer chain
    ind0 : int
        Index of first point from which to exponentially decaying probability
        with distance
    Returns
    -------
    int
        Index of new point selected based on distance from ind0
    """
    cdef long side
    cdef long window_side
    cdef long upper_bound

    if window > N_beads:
        raise ValueError(
            "Bead selection window size must be less than polymer length."
        )
    if window < 1:
        return ind0

    side = rand() % 2

    if side == 0:   # Select second bead on LHS
        window_side = max(min(window, ind0), 1)
        upper_bound = max(ind0, 1)
        return from_right(window_side, upper_bound)

    else:           # Select second bead on RHS
        window_side = max(min(window, N_beads - ind0), 1)
        upper_bound = max(N_beads - ind0, 1)
        return from_left(window_side, upper_bound) + ind0


cdef (long, long) check_bead_bounds(long bound_0, long bound_1, long num_beads):
    """Check the selected monomer bounds on the polymer.

    Correct invalid bounds and return starting and ending monomer indices.

    Parameters
    ----------
    bound_0, bound_1 : int
        First and second monomer bounds on the polymer.
    num_beads : int
        Number of monomeric units on the polymer.

    Returns
    -------
    Tuple[int, int]
        Starting and ending monomer indices after correcting for invalid bounds
    """
    cdef long ind0, indf

    bound_0 = min(bound_0, num_beads)
    bound_0 = max(bound_0, 0)

    if bound_1 > num_beads:
        ind0 = bound_0
        indf = num_beads
    elif bound_1 < 0:
        ind0 = 0
        indf = bound_0 + 1
    elif bound_0 == bound_1:
        ind0 = bound_0
        indf = ind0 + 1
    else:
        ind0 = min(bound_0, bound_1)
        indf = max(bound_0, bound_1)

    return ind0, indf

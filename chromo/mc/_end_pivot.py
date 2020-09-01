"""
Deterministic component of end-pivot move.

Given a set of points and a rotation magnitude, conduct the end-pivot move.

"""

import numpy as np

from chromo.mc.util.linalg import *

def conduct_end_pivot(r_points, r_pivot, r_base, t3_points, t2_points, rot_angle):
    """
    Rotation of fixed subset of beads

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing rotation
    r_pivot : array_like (4,)
        Homogeneous coordinates for bead about which the pivot occurs
    r_base : array_like (4,)
        Homogeneous coordinates for bead establishing axis of rotation
    t3_points : array_like (4, N)
        Homogeneous tangent vectors for beads undergoing rotation
    t2_points : array_like (4, N)
        Homogeneous tangent, orthogonal to t3 tangents, for rotating beads
    rot_angle : float
        Magnitude of counterclockwise rotation (in radians)
    
    Returns
    -------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following rotation
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    t2_trial : array_like (4, N)
        Homogeneous tangent vectors, orthogonal to t3_trial, following rotation

    """

    rot_matrix = arbitrary_axis_rotation(r_pivot, r_base, rot_angle)
    
    r_trial = rot_matrix @ r_points         # Generate trial coordinates
    t3_trial = rot_matrix @ t3_points       # Generate trial tangents
    t2_trial = rot_matrix @ t2_points       # Generate orthogonal trial tangents

    r_trial = r_trial[0:3, :].T
    t3_trial = t3_trial[0:3, :].T
    t2_trial = t2_trial[0:3, :].T

    return r_trial, t3_trial, t2_trial

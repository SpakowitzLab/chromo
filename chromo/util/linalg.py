"""
Utility functions for linear algebra calculations
"""
import random
import math as math

import numpy as np
from scipy.spatial.transform import Rotation as R


def arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle):
    """
    Generate a transformation matrix for counterclockwise rotation of angle rot_angle about an arbitrary axis from points r_ind0 to r_ind1.
    Parameters
    ----------
    r_ind0 : np.array (3 by 1)
        1D column vector of (x, y, z) coordinates for the first point forming the axis of rotation
    r_ind1 : np.array (3 by 1)
        1D column vector of (x, y, z) coordinates for the second point forming the axis of rotation
    rot_angle : float
        Magnitude of the angle of rotation about arbitrary axis
    Returns
    -------
    rot_mat : np.array (4 by 4)
        Homogeneous rotation matrix for rotation about arbitrary axis
    """
    
    # Move end of axis from r_ind0 to origin
    translate_mat = np.array([
        [1, 0, 0, -r_ind0[0]],
        [0, 1, 0, -r_ind0[1]],
        [0, 0, 1, -r_ind0[2]],
        [0, 0, 0, 1]
    ])

    inv_translation_mat = translate_mat.copy()
    inv_translation_mat[0:3, 3] *= -1

    dr = r_ind1 - r_ind0
    r = dr / np.linalg.norm(dr)
    
    rot_about_r = np.identity(4)
    rot_about_r[0,0] = np.cos(rot_angle) + r[0]**2*(1 - np.cos(rot_angle))
    rot_about_r[0,1] = r[0]*r[1]*(1 - np.cos(rot_angle)) - r[2]*np.sin(rot_angle)
    rot_about_r[0,2] = r[0]*r[2]*(1 - np.cos(rot_angle)) + r[1]*np.sin(rot_angle)
    rot_about_r[1,0] = r[0]*r[1]*(1 - np.cos(rot_angle)) + r[2]*np.sin(rot_angle)
    rot_about_r[1,1] = np.cos(rot_angle) + r[1]**2*(1 - np.cos(rot_angle))
    rot_about_r[1,2] = r[1]*r[2]*(1 - np.cos(rot_angle)) - r[0]*np.sin(rot_angle)
    rot_about_r[2,0] = r[0]*r[2]*(1 - np.cos(rot_angle)) - r[1]*np.sin(rot_angle)
    rot_about_r[2,1] = r[1]*r[2]*(1 - np.cos(rot_angle)) + r[0]*np.sin(rot_angle)
    rot_about_r[2,2] = np.cos(rot_angle) + r[2]**2*(1 - np.cos(rot_angle))

    rot_mat = inv_translation_mat @ rot_about_r @ translate_mat

    return rot_mat

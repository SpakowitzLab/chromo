"""
Utility functions for linear algebra calculations
"""
import random
import math as math

import numpy as np


def arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle):
    """
    Generate a transformation matrix for rotation of angle rot_angle about an arbitrary axis from points r_ind0 to r_ind1.

    Parameters
    ----------
    r_ind0:         (3, 1) np.array
                    1D column vector of (x, y, z) coordinates for the first point forming the axis of rotation

    r_ind1:         (3, 1) np.array
                    1D column vector of (x, y, z) coordinates for the second point forming the axis of rotation

    rot_angle:      float
                    Magnitude of the angle of rotation about arbitrary axis

    Returns
    -------
    rot_matrix:     (4, 4) np.array
                    Homogeneous rotation matrix for rotation about arbitrary axis

    """
    translate_mat = np.array([
		[1, 0, 0, -r_in1[0]],
		[0, 1, 0, -r_in1[1]],
		[0, 0, 1, -r_in1[2]],
		[0, 0, 0, 1]
	])

    inv_translation_mat = translation_mat
    inv_translation_mat[0:2, 3] *= -1

    # Calculate the length of the projections to point ind0 on yz plane.
    proj_len_yz = math.sqrt(r_ind0[1]**2 + r_ind0[2]**2)
    # rotate the vector ind0 into the xz plane
    rot_mat_x = np.identity(4)
    rot_mat_x[1:3, 1:3] = np.array([
        [r_ind0[2] / proj_len_yz, -r_ind0[1] / proj_len_yz],
        [r_ind0[1] / proj_len_yz, r_ind0[2] / proj_len_yz]
    ])
    # rotating by the same angle in the opposite direction is the inverse
    inv_rot_mat_x = rot_mat_x
    inv_rot_mat_x[1, 2] = -rot_mat_x[1, 2]
    inv_rot_mat_x[2, 1] = -rot_mat_x[2, 1]
    # now rotate around the y-axis such that the ind0 vector becomes the z-axis
    rot_mat_y = np.identity(4)
    rot_mat_y[0, 0] = proj_len_yz
    rot_mat_y[0, 2] = -r_ind0[0]
    rot_mat_y[2, 0] = r_ind0[0]
    rot_mat_y[2, 2] = proj_len_yz
    inv_rot_mat_y = rot_mat_y
    inv_rot_mat_y[0, 2] = -rot_mat_y[0, 2]
    inv_rot_mat_y[2, 0] = -rot_mat_y[2, 0]
    rot_mat_z = np.identity(4)  # now rotate about z
    rot_mat_z[0:1, 0:1] = np.array([
        [math.cos(rot_angle), -math.sin(rot_angle)]
        [-rot_mat_z[0, 1], rot_mat_z[0, 0]]
    ])
    rot_matrix = inv_translation_mat @ inv_rot_mat_x @ inv_rot_mat_y \
        @ rot_mat_z @ rot_mat_y @ rot_mat_x @ tranlate_mat
    return rot_matrix

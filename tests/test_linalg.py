# Test linalg

from chromo.util.linalg import *

import numpy as np

def test_arbitrary_axis_rotation():
    r_ind0 = np.array([0, 0, 1])
    r_ind_base = np.array([0, 0, 0])
    rot_angle = np.pi / 4

    rot_mat_1 = arbitrary_axis_rotation(r_ind0, r_ind_base, rot_angle)
    rot_mat_2 = arbitrary_axis_rotation(r_ind_base, r_ind0, rot_angle)

    expected_rot_mat = np.identity(4)
    expected_rot_mat[0:2, 0:2] = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)]])

    point_1 = np.array([1, 0, 0, 1])
    point_2 = np.array([np.cos(rot_angle), np.sin(rot_angle), 0, 1])    

    assert np.all(rot_mat_1 == expected_rot_mat)
    assert np.all(rot_mat_2 == expected_rot_mat)
    assert np.all(rot_mat_1 @ point_1 == point_2)
    assert np.all(rot_mat_2 @ point_1 == point_2)
    assert np.all(expected_rot_mat @ point_1 == point_2)

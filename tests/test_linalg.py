# Test linalg

from chromo.util.linalg import *

import numpy as np

def test_arbitrary_axis_rotation():
    r_ind1 = np.array([0, 0, 1]).T
    r_ind_base = np.array([0, 0, 0]).T
    rot_angle = np.pi / 4

    rot_mat_1 = arbitrary_axis_rotation(r_ind1, r_ind_base, rot_angle)
    rot_mat_2 = arbitrary_axis_rotation(r_ind_base, r_ind1, -rot_angle)
    
    print(rot_mat_1)
    print(rot_mat_2)

    expected_rot_mat_1 = np.identity(4)
    expected_rot_mat_1[0:2, 0:2] = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)]])

    expected_rot_mat_2 = np.identity(4)
    expected_rot_mat_2[0:2, 0:2] = np.array([[np.cos(-rot_angle), -np.sin(-rot_angle)],
        [np.sin(-rot_angle), np.cos(-rot_angle)]])
    
    assert np.all(np.isclose(rot_mat_1, rot_mat_2))
    assert np.all(rot_mat_1 == expected_rot_mat_2)
    assert np.all(rot_mat_2 == expected_rot_mat_2)

import chromo.util.linalg as clinalg

import numpy as np


def test_z_axis_rotation():
    r0 = np.array([0, 0, 0])
    r1 = np.array([0, 0, 1])
    rot_angle = np.pi / 4

    rot_mat = clinalg.arbitrary_axis_rotation(r0, r1, rot_angle)
    rot_mat_inv_inv = clinalg.arbitrary_axis_rotation(r1, r0, -rot_angle)

    expected_rot_mat = np.identity(4)
    expected_rot_mat[0:2, 0:2] = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)]
    ])

    assert np.all(np.isclose(rot_mat, rot_mat_inv_inv))
    assert np.all(np.isclose(rot_mat, expected_rot_mat))
    assert np.all(np.isclose(rot_mat_inv_inv, expected_rot_mat))

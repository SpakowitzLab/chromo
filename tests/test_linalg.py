# Test linalg

from chromo.util.linalg import *

import numpy as np

def test_arbitrary_axis_rotation():
	r_ind0 = np.array([0, 0, 1])
	r_ind1 = np.array([0, 0, 0])
	rot_angle = np.pi / 4

	rot_mat = arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle)

	expected_rot_mat = np.identity(4)
	expected_rot_mat[0:2, 0:2] = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
		[np.sin(rot_angle), np.cos(rot_angle)]])

	assert np.all(rot_mat == expected_rot_mat)

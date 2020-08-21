"""Utility functions for linear algebra calculations."""
import numpy as np
from scipy.spatial.transform import Rotation


def arbitrary_axis_rotation(r0, r1, rot_angle):
    """
    Rotate about an axis defined by two points.

    Generate a transformation matrix for counterclockwise (right handed
    convention) rotation of angle *rot_angle* about an arbitrary axis from
    points *r0* to *r1*.

    Parameters
    ----------
    r0 : (3,) array_like
        First point defining axis to rotate around.
    r1 : (3,) array_like
        Second point defining axis to rotate around.
    rot_angle : float
        Angle of rotation. Positive rotation is counterclockwise when the
        vector `r1 - r0` is pointing directly out of the blackboard.

    Returns
    -------
    rot_mat : (4, 4) array_like
        Homogeneous rotation matrix.
    """
    translate_mat = np.array([
        [1, 0, 0, -r0[0]],
        [0, 1, 0, -r0[1]],
        [0, 0, 1, -r0[2]],
        [0, 0, 0, 1]
    ])
    inv_translation_mat = translate_mat.copy()
    inv_translation_mat[0:3, 3] *= -1

    rot_axis = r1 - r0
    rot_vec = rot_angle * rot_axis/np.linalg.norm(rot_axis)
    rot_mat = np.identity(4)
    rot_mat[:3, :3] = Rotation.from_rotvec(rot_vec).as_matrix()

    return inv_translation_mat @ rot_mat @ translate_mat

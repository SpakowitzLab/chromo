"""Test Implementation of Nucleosome Geometry.
"""

import numpy as np
from wlcstat.chromo import OmegaE2E
from chromo.util.nucleo_geom import get_exiting_orientations, s_default
from chromo.util.linalg import get_rotation_matrix


def test_R_entry_exit():
    """Test computation of the nucleosome entry/exit rotation matrix.

    Notes
    -----
    The entry/exit rotation matrix provides the transformation matrix that
    converts from the entering DNA orientation to the exiting DNA orientation.
    """

    # Obtain the entry/exit rotation matrix from wlcstat
    R_wlcstat = OmegaE2E()

    # Obtain the rotation matrix from the nucleosome geometry module
    # Start by defining some arbitrary entering orientation
    T3_enter = np.array([1, 0, 0])
    T1_enter = np.array([0, 1, 0])
    T2_enter = np.cross(T3_enter, T1_enter)
    # Now get the exiting orientations
    T3_exit, T2_exit, T1_exit = get_exiting_orientations(
        s=s_default, T3_enter=T3_enter, T2_enter=T2_enter, T1_enter=T1_enter
    )
    # Now get the rotation matrix
    R_chromo = get_rotation_matrix(T3_enter, T2_enter, T3_exit, T2_exit)
    assert np.allclose(np.dot(R_chromo, T3_enter), T3_exit), \
        "R_chromo is incorrect. T3_enter does not map to T3_exit."
    assert np.allclose(np.dot(R_chromo, T3_enter), T3_exit), \
        "R_chromo is incorrect. T2_enter does not map to T2_exit."

    # Check if the two rotation matrices are the same
    test_points = np.random.rand(10, 3)  # Generate random 3D points
    transformed_points_1 = np.dot(R_wlcstat, test_points.T).T
    transformed_points_2 = np.dot(R_chromo, test_points.T).T

    # Check the angles between the transformed points
    angles_wlcstat = np.array([
        np.arctan2(  # Rotation about x-axis
            R_wlcstat[2, 1], R_wlcstat[2, 2]
        ),
        np.arctan2(  # Rotation about y-axis
            -R_wlcstat[2, 0], np.sqrt(R_wlcstat[2, 1]**2 + R_wlcstat[2, 2]**2)
        ),
        np.arctan2(  # Rotation about z-axis
            R_wlcstat[1, 0], R_wlcstat[0, 0]
        )
    ])
    angles_chromo = np.array([
        np.arctan2(  # Rotation about x-axis
            R_chromo[2, 1], R_chromo[2, 2]
        ),
        np.arctan2(  # Rotation about y-axis
            -R_chromo[2, 0], np.sqrt(R_chromo[2, 1]**2 + R_chromo[2, 2]**2)
        ),
        np.arctan2(  # Rotation about z-axis
            R_chromo[1, 0], R_chromo[0, 0]
        )
    ])

    # Print Rotation matrices and points for debugging
    print("Rotation Matrix from WLCSTAT:")
    print(R_wlcstat)
    print()
    print("Rotation Matrix from Chromo:")
    print(R_chromo)
    print()
    print("Original Points:")
    print(test_points)
    print()
    print("Transformed Points from WLCSTAT:")
    print(transformed_points_1)
    print()
    print("Transformed Points from Chromo:")
    print(transformed_points_2)
    print()
    print("Angles from WLCSTAT:")
    print(angles_wlcstat)
    print()
    print("Angles from Chromo:")
    print(angles_chromo)

    assert np.allclose(transformed_points_1, transformed_points_2), \
        "The two rotation matrices are not the same."
    assert np.allclose(angles_wlcstat, angles_chromo), \
        "The two rotation matrices produce inconsistent angles."


if __name__ == "__main__":
    test_R_entry_exit()

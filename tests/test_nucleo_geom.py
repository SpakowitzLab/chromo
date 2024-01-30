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

    print("Rotation Matrix from WLCSTAT:")
    print(R_wlcstat)
    print()
    print("Rotation Matrix from Chromo:")
    print(R_chromo)

    assert np.allclose(R_wlcstat, R_chromo)


if __name__ == "__main__":
    test_R_entry_exit()

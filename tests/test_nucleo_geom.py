"""Test Implementation of Nucleosome Geometry.
"""

import numpy as np
from chromo.util.nucleo_geom import get_exiting_orientations, s_default, \
    consts_dict, t3, t1, t2, normal, get_r, default_rot_matrix, \
    compute_exit_orientations_with_default_wrapping
from chromo.util.linalg import get_rotation_matrix


def test_binormal():
    """Test computation of the nucleosome binormal.
    """
    s = s_default
    T3_vec = t3(s, consts_dict)
    normal_vec = normal(s, consts_dict)
    binormal_vec = np.cross(T3_vec, normal_vec)
    assert np.allclose(np.linalg.norm(binormal_vec), 1), \
        "Binormal vector should have unit norm."
    assert np.allclose(np.linalg.norm(T3_vec), 1), \
        "T3 vector should have unit norm."
    assert np.allclose(np.linalg.norm(normal_vec), 1), \
        "Normal vector should have unit norm."
    assert np.allclose(np.dot(T3_vec, binormal_vec), 0), \
        "Binormal vector should be orthogonal to T3 vector."
    assert np.allclose(np.dot(normal_vec, binormal_vec), 0), \
        "Binormal vector should be orthogonal to normal vector."
    assert np.allclose(np.dot(T3_vec, normal_vec), 0), \
        "T3 and normal vectors should be orthogonal."
    R = consts_dict["R"]
    Lt = consts_dict["Lt"]
    h = consts_dict["h"]
    # The binormal is described analytically as:
    binormal_check = np.array([
        h / Lt * np.sin(2 * np.pi * s / Lt),
        -h / Lt * np.cos(2 * np.pi * s / Lt),
        2 * np.pi * R / Lt
    ])
    assert np.allclose(binormal_vec, binormal_check), \
        "Binormal not consistent with theory."


def test_T2_exit():
    """Test computation of the exiting T2 vector.
    """
    # Define some arbitrary entering orientation
    T3_enter = t3(0)
    T1_enter = t1(0)
    T2_enter = np.cross(T3_enter, T1_enter)
    # Get the exiting orientations
    _, T2_exit, _ = get_exiting_orientations(
        s=s_default, T3_enter=T3_enter, T2_enter=T2_enter, T1_enter=T1_enter
    )
    # Compute the T2 exit vector another way:
    T2_exit_2 = (
        np.dot(t2(s=s_default), t3(s=0)) * T3_enter +
        np.dot(t2(s=s_default), t2(s=0)) * T2_enter +
        np.dot(t2(s=s_default), t1(s=0)) * T1_enter
    )
    print("T2_exit: ", T2_exit)
    print("T2_exit_2: ", T2_exit_2)
    assert np.allclose(T2_exit, T2_exit_2), \
        "Error in computation of T2 vector."


def test_get_rotation_matrix():
    """Verify rotation matrix works between arbitrary vectors.

    Notes
    -----
    The `get_rotation_matrix` function in the `linalg` module is used to
    generate a rotation matrix that rotates two orthogonal vectors A and B
    onto two other orthogonal vectors A' and B'. The test checks that the
    method correctly performs the expected rotation.
    """
    # Define entering and exiting DNA tangents in the local frame
    t3_0 = t3(0)
    t1_0 = t1(0)
    t2_0 = np.cross(t3_0, t1_0)
    t3_1 = t3(s_default)
    t1_1 = t1(s_default)
    t2_1 = np.cross(t3_1, t1_1)
    # Compute the rotation matrix that converts the entering orientation to the
    # exiting orientation in the local frame
    R_local = get_rotation_matrix(t3_0, t1_0, t3_1, t1_1)
    # Rotate the entering DNA tangents to the exiting DNA tangents in the local
    # frame
    t3_1_check = np.dot(R_local, t3_0)
    t1_1_check = np.dot(R_local, t1_0)
    t2_1_check = np.dot(R_local, t2_0)
    # Check that the rotated vectors are consistent with the exiting DNA
    # tangents in the local frame
    assert np.allclose(t3_1, t3_1_check), \
        "Error rotating t3 vector using `get_rotation_matrix` in local frame."
    assert np.allclose(t1_1, t1_1_check), \
        "Error rotating t1 vector using `get_rotation_matrix` in local frame."
    assert np.allclose(t2_1, t2_1_check), \
        "Error rotating t2 vector using `get_rotation_matrix` in local frame."
    # Generate new entry tangents arbitrarily in space
    T3_entry = np.array([1, 0, 0])
    T1_entry = np.array([0, 1, 0])
    T2_entry = np.cross(T3_entry, T1_entry)
    # Compute the exit tangents associated with the arbitrary entry tangents
    T3_exit, T2_exit, T1_exit = get_exiting_orientations(
        s=s_default, T3_enter=T3_entry, T2_enter=T2_entry, T1_enter=T1_entry
    )
    # Compute the rotation matrix that converts the arbitrary entry tangents to
    # the computed exit tangents in the global frame
    R_global = get_rotation_matrix(T3_entry, T2_entry, T3_exit, T2_exit)
    # Compute the exiting DNA orientations in the global frame using `R`
    T3_exit_check = np.dot(R_global, T3_entry)
    T2_exit_check = np.dot(R_global, T2_entry)
    T1_exit_check = np.dot(R_global, T1_entry)
    # Check that the computed exit tangents in the global frame are consistent
    assert np.allclose(T3_exit, T3_exit_check), \
        "Error obtaining T3_exit using `get_rotation_matrix` in global frame."
    assert np.allclose(T2_exit, T2_exit_check), \
        "Error obtaining T2_exit using `get_rotation_matrix` in global frame."
    assert np.allclose(T1_exit, T1_exit_check), \
        "Error obtaining T1_exit using `get_rotation_matrix` in global frame."


def test_dot_product_matrix():
    """Test that entry/exit tangents produce the same dot product matrix.

    Notes
    -----
    The dot products between entering and exiting DNA tangents should produce
    the same dot product matrix regardless of reference frame. This is because
    the nucleosome is a fixed body -- the relationship between the entering
    and exiting DNA tangents is invariant to the reference frame. This
    relationship is captured by a 3x3 matrix produced from dot products of
    different combinations of entering and exiting DNA tangents. This test
    verifies that the dot product matrix is consistent in two reference frames.
    """
    # Define entering and exiting DNA tangents in the local frame
    t3_0 = t3(0)
    t1_0 = t1(0)
    t2_0 = np.cross(t3_0, t1_0)
    t3_1 = t3(s_default)
    t1_1 = t1(s_default)
    t2_1 = np.cross(t3_1, t1_1)
    # Generate dot product matrix in the local frame
    A_local = np.array([
        [np.dot(t1_1, t1_0), np.dot(t1_1, t2_0), np.dot(t1_1, t3_0)],
        [np.dot(t2_1, t1_0), np.dot(t2_1, t2_0), np.dot(t2_1, t3_0)],
        [np.dot(t3_1, t1_0), np.dot(t3_1, t2_0), np.dot(t3_1, t3_0)]
    ])
    # Generate new entry tangents arbitrarily in space
    T3_0 = np.array([1, 0, 0])
    T1_0 = np.array([0, 1, 0])
    T2_0 = np.cross(T3_0, T1_0)
    # Compute the exit tangents associated with the arbitrary entry tangents
    T3_1, T2_1, T1_1 = get_exiting_orientations(
        s=s_default, T3_enter=T3_0, T2_enter=T2_0, T1_enter=T1_0
    )
    # Generate dot product matrix in the global frame
    A_global = np.array([
        [np.dot(T1_1, T1_0), np.dot(T1_1, T2_0), np.dot(T1_1, T3_0)],
        [np.dot(T2_1, T1_0), np.dot(T2_1, T2_0), np.dot(T2_1, T3_0)],
        [np.dot(T3_1, T1_0), np.dot(T3_1, T2_0), np.dot(T3_1, T3_0)]
    ])
    # Check that the dot product matrices are consistent
    assert np.allclose(A_local, A_global), \
        "Error obtaining dot product matrix using `get_exiting_orientations`."


def test_rotations_with_dot_products():
    """Verify that dot product matrix produces the correct exit orientations.

    Notes
    -----
    We calculate the exiting DNA orientations from the entering DNA orientations
    using the dot product matrix. The entry/exit relationship can also be
    captured by rotation matrices. We can compute the rotation matrix that
    rotates the entering DNA orientation to the exiting DNA orientation in the
    local frame. Call this rotation matrix R1. Then for an arbitrary entering
    DNA orientation, we can rotate the orientation onto the local frame using
    rotation matrix R2. We can then rotate the orientation onto the exiting
    frame using rotation matrix R1. And finally, we can rotate back to the
    global frame using the inverse of rotation matrix R2.

    In this test, we compare the global exiting DNA orientations computed using
    the dot product matrix with the global exiting DNA orientations computed
    using the rotation matrices. We verify that the two methods produce the same
    global exiting DNA orientations.
    """
    # Define entering and exiting DNA tangents in the local frame
    t3_0 = t3(0)
    t1_0 = t1(0)
    t2_0 = np.cross(t3_0, t1_0)
    t3_1 = t3(s_default)
    t1_1 = t1(s_default)
    t2_1 = np.cross(t3_1, t1_1)
    # Generate dot product matrix in the local frame
    R1 = get_rotation_matrix(t3_0, t2_0, t3_1, t2_1)
    # Generate new entry tangents arbitrarily in space
    T3_0 = np.array([1, 0, 0])
    T1_0 = np.array([0, 1, 0])
    T2_0 = np.cross(T3_0, T1_0)
    # Compute the rotation matrix that rotates onto the local frame
    R2 = get_rotation_matrix(T3_0, T2_0, t3_0, t2_0)
    # Compute the exit tangents associated with the arbitrary entry tangents
    # using the dot product matrix
    T3_1, T2_1, T1_1 = get_exiting_orientations(
        s=s_default, T3_enter=T3_0, T2_enter=T2_0, T1_enter=T1_0
    )
    # Compute the exit tangent associated with the arbitrary entry tangents
    # using the rotation matrices
    T3_1_check = np.dot(np.linalg.inv(R2), np.dot(R1, np.dot(R2, T3_0)))
    T2_1_check = np.dot(np.linalg.inv(R2), np.dot(R1, np.dot(R2, T2_0)))
    T1_1_check = np.dot(np.linalg.inv(R2), np.dot(R1, np.dot(R2, T1_0)))
    # Check that the exit tangents are consistent
    assert np.allclose(T3_1, T3_1_check), \
        "Error obtaining T3_exit using dot product matrix."
    assert np.allclose(T2_1, T2_1_check), \
        "Error obtaining T2_exit using dot product matrix."
    assert np.allclose(T1_1, T1_1_check), \
        "Error obtaining T1_exit using dot product matrix."


def test_transformation_of_r():
    """Verify relative entry/exit position is consistent in different frames.

    Notes
    -----
    There are two ways to compute the exit position given an entry position.
    One way is to compute the exit position in the local frame and then rotate
    the position to the global frame. This is how the transformation is done in
    the nucleo_geom module.

    Another way to compute the exiting orientation from an entry position is to
    first take the difference vector between the entry and exit positions in the
    local frame. Then compute the dot products of that difference vector with
    the local frame entry tangents. Call these dot products B = [b1, b2, b3].
    Next, take your entry position in the global frame and add b1 * T1_entry +
    b2 * T2_entry + b3 * T3_entry. This will give you the exit position in the
    global frame.

    This test verifies that the two methods produce the same exit position in
    the global frame.
    """
    # Define entering and exiting DNA orientations in the local frame
    t3_0 = t3(0)
    t1_0 = t1(0)
    t2_0 = np.cross(t3_0, t1_0)
    t3_1 = t3(s_default)
    t1_1 = t1(s_default)
    t2_1 = np.cross(t3_1, t1_1)
    # Compute the entry/exit positions in the local frame
    r_local = np.array([0, 0, 0])
    r0_local = get_r(s=0)
    r1_local = get_r(s=s_default)
    # Compute the coefficients B = [b1, b2, b3] using the local frame
    B0 = np.dot(r0_local - r_local, np.array([t1_0, t2_0, t3_0]))
    B = np.dot(r1_local - r0_local, np.array([t1_0, t2_0, t3_0]))
    # Verify that B can be used to compute the exit position in the local frame
    r1_local_check = r0_local + B[0] * t1_0 + B[1] * t2_0 + B[2] * t3_0
    assert np.allclose(r1_local, r1_local_check), \
        "Error obtaining r_exit using dot products in local frame."
    # Generate new orientations arbitrarily in space
    T3_0 = np.array([1, 0, 0])
    T1_0 = np.array([0, 1, 0])
    T2_0 = np.cross(T3_0, T1_0)
    # Generate a new entry position arbitrarily in space
    r_global = np.array([134, 234, 341])
    # Compute rotation matrix that converts the local frame to the global frame
    R_local_to_global = get_rotation_matrix(t3_0, t2_0, T3_0, T2_0)
    # Compute the entry/exit position using the rotation matrix
    r0_global = np.dot(R_local_to_global, r0_local) + r_global
    r1_global = np.dot(R_local_to_global, r1_local) + r_global
    # Compute the exit position using the coefficients B
    r0_global_check = r_global + B0[0] * T1_0 + B0[1] * T2_0 + B0[2] * T3_0
    r1_global_check = r0_global + B[0] * T1_0 + B[1] * T2_0 + B[2] * T3_0
    # Verify that the two methods produce same exit position in the global frame
    assert np.allclose(r0_global, r0_global_check), \
        "Error obtaining r_entry using dot products in global frame."
    assert np.allclose(r1_global, r1_global_check), \
        "Error obtaining r_exit using dot products in global frame."


def test_default_rotation_matrix():
    """Test the implementation of the default rotation matrix.
    """
    # Define arbitrary entry orientations
    T3_0 = np.array([1, 0, 0])
    T1_0 = np.array([0, 1, 0])
    T2_0 = np.cross(T3_0, T1_0)
    # Get the exit orientations using the `get_exiting_orientations` function
    T3_1, T2_1, T1_1 = get_exiting_orientations(
        s=s_default, T3_enter=T3_0, T2_enter=T2_0, T1_enter=T1_0
    )
    # Get the exiting orientations using the default rotation matrix
    T_in_matrix = np.column_stack((T1_0, T2_0, T3_0))
    T_out_matrix = np.dot(T_in_matrix, default_rot_matrix[:3].T)
    T1_1_check = T_out_matrix[:, 0]
    T2_1_check = T_out_matrix[:, 1]
    T3_1_check = T_out_matrix[:, 2]
    # Verify that the two methods produce the same exit orientations
    assert np.allclose(T1_1, T1_1_check), \
        "Error obtaining T1_exit using default rotation matrix."
    assert np.allclose(T2_1, T2_1_check), \
        "Error obtaining T2_exit using default rotation matrix."
    assert np.allclose(T3_1, T3_1_check), \
        "Error obtaining T3_exit using default rotation matrix."
    T3_1_check2, T2_1_check2, T1_1_check2 = \
        compute_exit_orientations_with_default_wrapping(T3_0, T2_0, T1_0)
    assert np.allclose(T1_1, T1_1_check2), \
        "Error obtaining T1_exit using default rotation matrix and function."
    assert np.allclose(T2_1, T2_1_check2), \
        "Error obtaining T2_exit using default rotation matrix and function."

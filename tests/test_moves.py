"""Test the "deterministic" part of each function for proposing MC moves.
"""

import pyximport
pyximport.install()

# External Modules
import numpy as np

# Custom Modules
import chromo.polymers as ply
import chromo.mc.move_funcs as mv
import chromo.util.linalg as la


def test_determinitic_end_pivot():
    """Test End pivot move w/ 90 deg clockwise rotation about positive x axis.

    A 90 degree clockwise rotation is equivalent to a -90 degree rotation about
    the x-axis or a 90 degree rotation about the -x axis.
    """
    axis = np.ascontiguousarray(np.array([1, 0, 0], dtype=float))
    rot_angle = - np.pi / 2
    c = np.sqrt(0.5)

    r_points = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, -1, 0, 1],
            [3, -2, 0, 1]
        ], dtype=float
    ))

    r_expected = np.ascontiguousarray(np.array(
        [
            [1, 0, 0, 1],
            [2, 0, 0, 1],
            [3, 0, 0, 1],
            [3, 0, 1, 1],
            [3, 0, 2, 1]
        ], dtype=float
    ))

    t3_points = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [c, -c, 0, 0],
            [0, -1, 0, 0],
            [0, -1, 0, 0]
        ], dtype=float
    ))

    t3_expected = np.ascontiguousarray(np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [c, 0, c, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ], dtype=float
    ))

    t2_points = np.ascontiguousarray(np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ], dtype=float
    ))

    t2_expected = np.ascontiguousarray(np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float
    ))
    rot_mat = la.return_arbitrary_axis_rotation(
        axis, r_points[0, 0:3], rot_angle
    )
    poly = ply.PolymerBase(
        name="polymer1", r=r_points, t3=t3_points, t2=t2_points
    )
    poly.transformation_mat = rot_mat
    inds = np.arange(5)
    n_inds = 5
    mv.transform_r_t3_t2(poly, inds, n_inds)
    r = np.asarray(poly.r_trial)
    t3 = np.asarray(poly.t3_trial)
    t2 = np.asarray(poly.t2_trial)

    assert np.all(np.isclose(r, r_expected))
    assert np.all(np.isclose(t3, t3_expected))
    assert np.all(np.isclose(t2, t2_expected))


def test_determinitic_slide_move():
    """Test deterministic component of slide move.
    """

    r_points = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, -1, 0, 1],
            [3, -2, 0, 1]
        ], dtype='d'
    ))

    translate_x = 1
    translate_y = 2
    translate_z = 3.5

    r_expected = np.ascontiguousarray(np.array(
        [
            [2, 2, 3.5, 1],
            [3, 2, 3.5, 1],
            [4, 2, 3.5, 1],
            [4, 1, 3.5, 1],
            [4, 0, 3.5, 1]
        ], dtype='d'
    ))

    poly = ply.PolymerBase(
        name="polymer1", r=r_points
    )
    la.generate_translation_mat(
        np.array([translate_x, translate_y, translate_z]), poly
    )
    inds = np.arange(5)
    n_inds = 5
    mv.transform_r_t3_t2(poly, inds, n_inds)
    r_observed = np.asarray(poly.r_trial)

    assert np.all(np.isclose(r_observed, r_expected))


def test_deterministic_tangent_rotation():
    """Test the deterministic component of the tangent rotation move.

    Rotate the tangent vectors (2 * pi / 3) degrees counterclockwise about an
    axis defined by [1, 1, 1] / sqrt(3).
    """
    r_point = np.ascontiguousarray(
        np.atleast_2d(np.array([1, 2, 3], dtype='d'))
    )
    t3_point = np.ascontiguousarray(
        np.atleast_2d(np.array([0, 0, 1], dtype='d'))
    )
    t2_point = np.ascontiguousarray(
        np.atleast_2d(np.array([0, 1, 0], dtype='d'))
    )

    axis = np.array([1, 1, 1]) / np.sqrt(3)
    rot_angle = 2 * np.pi / 3
    rot_mat = la.return_arbitrary_axis_rotation(axis, r_point[0], rot_angle)

    r_expected = np.ascontiguousarray(
        np.atleast_2d(np.array([1, 2, 3], dtype='d'))
    )
    t3_expected = np.ascontiguousarray(
        np.atleast_2d(np.array([1, 0, 0], dtype='d'))
    )
    t2_expected = np.ascontiguousarray(
        np.atleast_2d(np.array([0, 0, 1], dtype='d'))
    )

    poly = ply.PolymerBase(
        name="polymer1", r=r_point, t3=t3_point, t2=t2_point
    )
    poly.transformation_mat = rot_mat
    inds = np.arange(1)
    n_inds = 1
    mv.transform_r_t3_t2(poly, inds, n_inds)
    r = np.asarray(poly.r_trial)
    t3 = np.asarray(poly.t3_trial)
    t2 = np.asarray(poly.t2_trial)

    assert np.all(np.isclose(r, r_expected))
    assert np.all(np.isclose(t3, t3_expected))
    assert np.all(np.isclose(t2, t2_expected))


def test_deterministic_crank_shaft_move():
    """Test deterministic component of the crank-shaft move.
    """
    rot_angle = np.pi / 2
    c = np.sqrt(0.5)

    r_poly = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, -1, 0, 1],  # 3
            [3, -2, 0, 1],  # 4
            [4, -2, 0, 1],  # 5
            [5, -2, 0, 1],  # 6
            [5, -1, 0, 1],  # 7
            [5,  0, 0, 1],
            [6,  0, 0, 1],
            [7,  0, 0, 1]
        ], dtype='d'
    ))

    r_points = r_poly.copy()

    r_expected = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 1],
            [2,  0, 0, 1],
            [3,  0, 0, 1],
            [3, 0, 1, 1],   # 3
            [3, 0, 2, 1],   # 4
            [4, 0, 2, 1],   # 5
            [5, 0, 2, 1],   # 6
            [5, 0, 1, 1],   # 7
            [5,  0, 0, 1],
            [6,  0, 0, 1],
            [7,  0, 0, 1]
        ], dtype='d'
    ))

    t3_poly = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [c, -c, 0, 0],
            [0, -1, 0, 0],  # 3
            [1,  0, 0, 0],  # 4
            [1,  0, 0, 0],  # 5
            [-c, c, 0, 0],  # 6
            [0,  1, 0, 0],  # 7
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [1,  0, 0, 0]
        ], dtype='d'
    ))

    t3_points = t3_poly.copy()

    t3_expected = np.ascontiguousarray(np.array(
        [
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [c, -c, 0, 0],
            [0,  0, 1,  0],  # 3
            [1,  0, 0,  0],  # 4
            [1,  0, 0,  0],  # 5
            [-c, 0, -c, 0],  # 6
            [0,  0, -1, 0],  # 7
            [1,  0, 0, 0],
            [1,  0, 0, 0],
            [1,  0, 0, 0]
        ], dtype='d'
    ))

    axis = r_poly[2, 0:3] - r_poly[8, 0:3]
    axis = axis / np.linalg.norm(axis)
    point = r_points[0, 0:3].flatten()
    rot_mat = la.return_arbitrary_axis_rotation(axis, point, rot_angle)

    poly = ply.PolymerBase(
        name="polymer1", r=r_points, t3=t3_points
    )
    poly.transformation_mat = rot_mat
    inds = np.arange(3, 8)
    n_inds = 5
    mv.transform_r_t3_t2(poly, inds, n_inds)
    r = np.asarray(poly.r_trial)
    t3 = np.asarray(poly.t3_trial)

    assert np.all(np.isclose(r, r_expected))
    assert np.all(np.isclose(t3, t3_expected))

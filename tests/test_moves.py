"""
Test the "deterministic" part of each function for proposing MC moves.
"""
import numpy as np

from chromo.mc.moves import *

def test_determinitic_end_pivot():
    # pivot 90 degrees around the positive x axis
    r_base = np.array([0, 0, 0])
    r_pivot = np.array([1, 0, 0])
    rot_angle = np.pi / 2
    c = np.sqrt(0.5)

    r_points = np.array([   [1,  0, 0, 1],
                            [2,  0, 0, 1],
                            [3,  0, 0, 1],
                            [3, -1, 0, 1],
                            [3, -2, 0, 1]]).T

    r_expected = np.array([ [1, 0, 0, 1],
                            [2, 0, 0, 1],
                            [3, 0, 0, 1],
                            [3, 0, 1, 1],
                            [3, 0, 2, 1]]).T

    t3_points = np.array([  [1,  0, 0, 1],
                            [1,  0, 0, 1],
                            [c, -c, 0, 1],
                            [0, -1, 0, 1],
                            [0, -1, 0, 1]]).T

    t3_expected = np.array([[1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [c, 0, c, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1]]).T

    t2_points = np.array([  [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1]]).T

    t2_expected = np.array([[0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 1, 0, 1]]).T

    r, t3, t2 = conduct_end_pivot(r_points, r_pivot, r_base, t3_points, t2_points,
        rot_angle)

    assert np.all(np.isclose(r, r_expected))
    assert np.all(np.isclose(t3, t3_expected))
    assert np.all(np.isclose(t2, t2_expected))


def test_determinitic_slide_move():

    r_points = np.array([   [1,  0, 0, 1],
                            [2,  0, 0, 1],
                            [3,  0, 0, 1],
                            [3, -1, 0, 1],
                            [3, -2, 0, 1]]).T

    translate_x = 1
    translate_y = 2
    translate_z = 3.5

    r_expected = np.array([ [2, 2, 3.5, 1],
                            [3, 2, 3.5, 1],
                            [4, 2, 3.5, 1],
                            [4, 1, 3.5, 1],
                            [4, 0, 3.5, 1]]).T

    r_observed = conduct_slide(r_points, translate_x, translate_y, translate_z)

    assert np.all(np.isclose(r_observed, r_expected))


def test_deterministic_tangent_rotation():
    """ Test the deterministic component of the tangent rotation move """

    r_point = np.array([0, 0, 0, 1])
    t3_point = np.array([0, 0, 1, 1])
    t2_point = np.array([0, 1, 0, 1])

    phi = np.pi / 2
    theta = np.pi / 4
    rot_angle = np.pi / 4

    t3_expected = np.array([0.5, 0.14644661, 0.85355339, 1])
    t2_expected = np.array([-0.5, 0.85355339, 0.14644661, 1])

    t3_observed, t2_observed = conduct_tangent_rotation(r_point,
        t3_point, t2_point, phi, theta, rot_angle)

    assert np.all(np.isclose(t3_observed, t3_expected))
    assert np.all(np.isclose(t2_observed, t2_expected))



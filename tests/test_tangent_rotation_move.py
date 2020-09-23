
import numpy as np

from chromo.mc.moves import *

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




import numpy as np

from chromo.mc.moves import *

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

    r_observed = conduct_slide_move(r_points, translate_x, translate_y, translate_z)

    assert np.all(np.isclose(r_observed, r_expected))


import numpy as np

from integration.mc import *
from chromo.mc._end_pivot import *

def test_determinitic_end_pivot():

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


def test_end_pivot_move():
    test = andy_mc_end_pivot()     
    
    assert True


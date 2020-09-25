"""
MCmove class

Creates a Monte Carlo move object that contains parameters for the particular
move.
"""

import numpy as np

from chromo.util.bead_selection import *
from chromo.util.linalg import *

### ---------- Classes ----------###

class MCAdapter:
    """
    Track success rate and adjust parameters for a Monte Carlo move.

    In order to ensure that a Monte Carlo simulation equilibrates as quickly as
    possible, we provide an automated system for tuning the "aggressiveness" of
    each Monte Carlo move in order to optimize the tradeoff between

    1. The move being large enough that it does not take too many moves to
       equilibrate, and
    2. The move not being so large that it will never be
       accepted, or the computational cost becomes prohibitive (especially in
       the case when the computational cost scales super-linearly).

    All Monte-Carlo moves we currently use for our polymers have two
    "amplitudes".  The "bead" amplitude controls how many beads are typically
    affected by the move, and the "move" amplitude controls how "far" those
    beads are moved.  Typically, actually bead counts and move sizes are
    selected randomly, so these amplitudes will correspond to averages.

    This module keeps track of the success rate (``num_attempt`` vs.
    ``num_success``) in order to dynamically adjust these amplitudes throughout
    the simulation to ensure they maintain reasonable values.

    In the future, we want to do this optimization by looking at actual metrics
    based on the simulation output, but for now only implementation of an
    MCAdapter (`MCAdaptSuccessRate`) handles this optimization by simply trying
    to maintain a fixed target success rate.
    """
    def __init__(self, move_func):
        self.name = move_func.__name__
        self.move_func = move_func
        self.amp_move = 2 * np.pi
        self.num_per_cycle = 1
        self.amp_bead = 10
        self.num_attempt = 0
        self.num_success = 0
        self.move_on = True

    def __str__(self):
        return f"MCAdapter<{name}>"

    def propose(self, polymer):
        """
        Get new proposed state of the system.

        Parameters
        ----------
        polymer : `chromo.Polymer`
            The polymer to attempt to move.

        Returns
        -------
        ind0 : int
            The first bead index to be moved.
        indf : int
            The last bead index to be moved.
        r : (N, 3) array_like of float, optional
            The proposed new positions of the moved beads. Throughout this
            method, ``N = indf - ind0 + 1`` is the number of moved beads.
        t3 : (N, 3) array_like of float, optional
            The proposed new tangent vectors.
        t2 : (N, 3) array_like of float, optional
            Proposed new material normals.
        states : (N, M) array_like of int, optional
            The proposed new chemical states, where *M* is the number of
            chemical states associated with the given polymer.
        """
        self.num_attempt += 1
        return self.move_func(polymer=polymer, amp_move=self.amp_move,
                              amp_bead=self.amp_bead)

    def accept(self):
        self.num_success += 1


### ---------- MC Move Functions ---------- ###

def crank_shaft_move(polymer, amp_move, amp_bead):
    """
    Rotate section of polymer around axis formed by two bounding beads.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (rotation angle) of the end-pivot move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the end-pivot move
    
    """
    print("CRANK-SHAFT MOVE")

    # Select ind0 and indf for the crank-shaft move
    delta_ind = min(np.random.randint(2, amp_bead), polymer.num_beads)
    ind0 = np.random.randint(polymer.num_beads - delta_ind + 1)
    indf = ind0 + delta_ind

    # Generate the rotation matrix and vector around the vector between bead ind0 and indf
    rot_angle = amp_move * (np.random.rand() - 0.5)

    if ind0 == (indf + 1):
        delta_t3 = polymer.t3[ind0, :]
    else:
        delta_t3 = polymer.r[indf - 1, :] - polymer.r[ind0, :]
        delta_t3 /= np.linalg.norm(delta_t3)

    r_ind0 = polymer.r[ind0, :]

    rot_matrix = np.zeros((3, 3), 'd')

    rot_matrix[0, 0] = delta_t3[0] ** 2. + (delta_t3[1] ** 2 + delta_t3[2] ** 2) * np.cos(rot_angle)
    rot_matrix[0, 1] = delta_t3[0] * delta_t3[1] * (1 - np.cos(rot_angle)) - delta_t3[2] * np.sin(rot_angle)
    rot_matrix[0, 2] = delta_t3[0] * delta_t3[2] * (1 - np.cos(rot_angle)) + delta_t3[1] * np.sin(rot_angle)

    rot_matrix[1, 0] = delta_t3[0] * delta_t3[1] * (1 - np.cos(rot_angle)) + delta_t3[2] * np.sin(rot_angle)
    rot_matrix[1, 1] = delta_t3[1] ** 2. + (delta_t3[0] ** 2 + delta_t3[2] ** 2) * np.cos(rot_angle)
    rot_matrix[1, 2] = delta_t3[1] * delta_t3[2] * (1 - np.cos(rot_angle)) - delta_t3[0] * np.sin(rot_angle)

    rot_matrix[2, 0] = delta_t3[0] * delta_t3[2] * (1 - np.cos(rot_angle)) - delta_t3[1] * np.sin(rot_angle)
    rot_matrix[2, 1] = delta_t3[1] * delta_t3[2] * (1 - np.cos(rot_angle)) + delta_t3[0] * np.sin(rot_angle)
    rot_matrix[2, 2] = delta_t3[2] ** 2. + (delta_t3[0] ** 2 + delta_t3[1] ** 2) * np.cos(rot_angle)

    rot_vector = np.cross(r_ind0, delta_t3) * np.sin(rot_angle)
    rot_vector[0] += (r_ind0[0] * (1 - delta_t3[0] ** 2)
                      - delta_t3[0]*(r_ind0[1] * delta_t3[1] + r_ind0[2] * delta_t3[2])) * (1 - np.cos(rot_angle))
    rot_vector[1] += (r_ind0[1] * (1 - delta_t3[1] ** 2)
                      - delta_t3[1]*(r_ind0[0] * delta_t3[0] + r_ind0[2] * delta_t3[2])) * (1 - np.cos(rot_angle))
    rot_vector[2] += (r_ind0[2] * (1 - delta_t3[2] ** 2)
                      - delta_t3[2]*(r_ind0[0] * delta_t3[0] + r_ind0[1] * delta_t3[1])) * (1 - np.cos(rot_angle))

    # Generate the trial positions and orientations
    r_poly_trial = np.zeros((indf - ind0, 3), 'd')
    t3_poly_trial = np.zeros((indf - ind0, 3), 'd')
    for i_bead in range(ind0, indf):
        r_poly_trial[i_bead - ind0, :] = rot_vector + np.matmul(rot_matrix, polymer.r[i_bead, :])
        t3_poly_trial[i_bead - ind0, :] = np.matmul(rot_matrix, polymer.t3[i_bead, :])
    
    return ind0, indf, r_poly_trial, t3_poly_trial, [None], [None]


def conduct_end_pivot(r_points, r_pivot, r_base, t3_points, t2_points, rot_angle):
    """
    Rotation of fixed sub set of beads.

    Deterministic component of end-pivot move.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing rotation
    r_pivot : array_like (4,)
        Homogeneous coordinates for beads about which the pivot occurs
    r_base : array_like (4,)
        Homogeneous coordinates for bead establishing axis of rotation
    t3_points : array_like (4, N)
        Homogeneous tangent vectors for beads undergoing rotation
    t2_points : array_like (4, N)
        Homogeneous tangent, orthogonal to t3 tangents, for rotating beads
    rot_angle : float
        Magnitude of counterclockwise rotation (in radians)

    Returns
    -------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following rotation
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    t2_trial : array_like (4, N)
        Homogeneous tangent vectors, orthogonal to t3_trial, following rotation
    
    """

    rot_matrix = arbitrary_axis_rotation(r_pivot, r_base, rot_angle)

    r_trial = rot_matrix @ r_points     # Generate trial copordinates
    t3_trial = rot_matrix @ t3_points   # Generate trial tangents
    t2_trial = rot_matrix @ t2_points   # Generate orthogonal trial tangents

    return r_trial, t3_trial, t2_trial


def end_pivot_move(polymer, amp_move, amp_bead):
    """
    Randomly rotate segment from one end of the polymer.

    Stochastic component of end-pivot move.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (rotation angle) of the end-pivot move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the end-pivot move

    """

    print("END-PIVOT MOVE")

    # Select a rotation angle based on the move amplitude
    rot_angle = amp_move * (np.random.rand() - 0.5)

    # isolate the number of beads in the polymer
    num_beads = polymer.num_beads
 
    # Randomly select beads
    if np.random.randint(0, 2) == 0:    # rotate LHS of polymer
        ind0 = 0
        pivot_point = indf = select_bead_from_left(amp_bead-1, num_beads)
        base_point = pivot_point + 1
    else:                               # rotate RHS of polymer
        pivot_point = ind0 = select_bead_from_right(amp_bead-1, num_beads)
        indf = num_beads-1
        base_point = pivot_point - 1

    # Isolate homogeneous coordinates and orientation for move
    r_points = np.ones((4, indf-ind0+1))
    r_pivot = polymer.r[pivot_point, :]
    r_base = polymer.r[base_point, :]
    t3_points = np.ones((4, indf-ind0+1))
    t2_points = np.ones((4, indf-ind0+1))

    # Reformat coordinates and orientation for matrix multiplication
    r_points[0:3, :] = polymer.r[ind0:indf+1, :].T
    t3_points[0:3, :] = polymer.t3[ind0:indf+1, :].T
    t2_points[0:3, :] = polymer.t2[ind0:indf+1, :].T

    # Conduct the end-pivot move and output new homogeneous coordinates/orientations
    r_trial, t3_trial, t2_trial = conduct_end_pivot(r_points, r_pivot, r_base, 
        t3_points, t2_points, rot_angle)   
    
    # Reformat from homogeneous to cartesian coordinates
    r_trial = r_trial[0:3, :].T
    t3_trial = t3_trial[0:3, :].T
    t2_trial = t2_trial[0:3, :].T
    return ind0, indf+1, r_trial, t3_trial, t2_trial, polymer.states[ind0:indf+1]


def conduct_slide_move(r_points, translation_x, translation_y, translation_z):
    """
    Conduct the slide move on set of beads with homogeneous coordinates stored in r_points.

    Deterministic component of slide move.

    Parameters
    ----------
    r_points : array_like (4, N)
        Homogeneous coordinates for beads undergoing slide move
    translation_x : float
        Translation in the x-direction
    translation_y : float
        Translation in the y-direction
    translation_z : float
        Translation in the z-direction

    Return
    ------
    r_trial : array_like (4, N)
        Homogeneous coordinates of beads following slide move
    
    """

    translation_mat = generate_translation_mat(translation_x, translation_y, translation_z)
    r_trial = translation_mat @ r_points

    return r_trial


def slide_move(polymer, amp_move, amp_bead):
    """
    Randomly translate a segment of the polymer.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Maximum amplitude (translation distance) of the slide move
    amp_bead : int
        Maximum amplitude (number) of beads affected by the slide move

    """
    print("SLIDE MOVE")

    # Set the translation distance based on slide amplitude
    translation_amp = amp_move * (np.random.rand())

    # Randomly partition translation move into x, y, z components
    rand_z = np.random.rand()
    rand_angle = np.random.rand() * 2 * np.pi
    translation_z = translation_amp * rand_z
    translation_y = np.sqrt(1 - rand_z**2) * translation_amp * np.sin(rand_angle)
    translation_x = np.sqrt(1 - rand_z**2) * translation_amp * np.cos(rand_angle)

    # Select a segment of nucleosomes on which to apply the move
    num_beads = polymer.num_beads
    ind0 = np.random.randint(num_beads)
    select_window = np.amin(np.array([num_beads - ind0, ind0, amp_bead]))
    indf = select_bead_from_point(select_window, num_beads, ind0)

    # Verify indices have integer variable types
    ind0 = int(ind0)
    indf = int(indf)
    
    # Generate matrix of homogeneous coordinates for beads undergoing slide move
    print(ind0)
    print(indf)
    r_points = np.ones((4, indf-ind0 + 1))
    r_points[0:3, :] = polymer.r[ind0:indf+1, :].T

    # Generate trial coordinates
    r_trial = conduct_slide_move(r_points, translation_x, translation_y, translation_z)
    r_trial = r_trial[0:3, :].T

    return ind0, indf, r_trial, polymer.t3[ind0:indf+1], polymer.t2[ind0:indf+1], polymer.states[ind0:indf+1]


def conduct_tangent_rotation(r_point, t3_point, t2_point, phi, theta, rot_angle):
    """
    Conduct the tangent rotation  move on set of beads with known (homogeneous) orientation vectors

    Deterministic component of tangent rotation move.

    Parameters
    ----------
    r_point : array_like (4,)
        Homogeneous coordinates for bead undergoing rotation
    t3_point : array_like (4,)
        Homogeneous tangent vectors for bead undergoing rotation
    t2_point : array_like (4,)
        Homogeneous tangent, orthogonal to t3 tangent, for rotating bead
    phi : float
        Magnitude of counterclockwise rotation (in radians) from x-axis in the xy plane defining axis of rotation
    theta : float
        Magnitude of angle from the positive z-axis defining axis of rotation
    rot_angle : float
        Magnitude of rotation about which to rotate bead
    
    Returns
    -------
    t3_trial : array_like (4, N)
        Homogeneous tangent vectors for beads following rotation
    t2_trial : array_like (4, N)
        Homogeneous tangent vectors, orthogonal to t3 tangents, following rotation

    """
    
    # Generate an arbitrary unit axis about which to conduct rotation
    r_ind0 = r_point
    r = 1
    del_x = r * np.sin(theta) * np.cos(phi)
    del_y = r * np.sin(theta) * np.sin(phi)
    del_z = r * np.cos(theta)
    r_ind1 = r_ind0 + np.array([del_x, del_y, del_z, 0])

    # Generate rotation matrix around axis connecting r_ind0 and r_ind1
    rot_matrix = arbitrary_axis_rotation(r_ind0[0:3], r_ind1[0:3], rot_angle)
    
    # Rotate tangent vectors using the rotation matrix
    t3_trial = rot_matrix @ t3_point
    t2_trial = rot_matrix @ t2_point
    
    return t3_trial, t2_trial

def one_bead_tangent_rotation_move(polymer, amp_move, num_beads):
    """
    Random bead rotation for a single bead.

    Stochastic component of random tangent vector rotations.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Fraction of 2 * pi maximum rotation to rotate bead (between 0-1)
    num_beads : int
        Number of beads in the polymer undergoing random bead rotations

    """
    
    # Generate a random amplitude for the rotation
    rot_angle = amp_move * np.random.uniform(0, 2 * np.pi)

    # Randomly select a bead in the polymer
    ind0 = np.random.randint(num_beads)
    r_point = polymer.r[ind0, :]
    t3_point = polymer.t3[ind0, :]
    t2_point = polymer.t2[ind0, :]

    # Select an arbitrary axis about which to conduct rotation
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)

    # Generate trial orientations for the beads undergoing rotation
    t3_trial, t2_trial = conduct_tangent_rotation(r_point, t3_point, t2_point, phi, theta, rot_angle)

    return ind0, polymer.num_beads, polymer.r, t3_trial, t2_trial, polymer.states[ind0, :]


def tangent_rotation_move(polymer, amp_move, amp_bead):
    """
    Random bead rotation for a random selection of beads.

    Calls *one_bead_tangent_rotation_move* for each randomly selected bead to undergo rotation.

    Parameters
    ----------
    polymer : Polymer
        Polymer object
    amp_move : float
        Fraction of maximum rotation space to rotate beads (between 0-1)
        Scales spherical angles to a random value within a fraction of their maximum values
    amp_bead : int
        Number of beads to randomly rotate

    """

    print("TANGENT ROTATION MOVE")
    
    # Select some number of beads to undergo rotation
    num_beads_to_move = np.random.uniform() * amp_beads

    # Identify the number of beads in the polymer
    num_beads = polymer.num_beads

    # Initialize index vector pointing to rotated beads
    adjusted_beads = []

    # Initialize lists of trial orientations for rotated beads
    all_t3_trial = []
    all_t2_trial = []

    # Loop through number of beads being moved, store the indices and trial orientations of each bead
    for i in range(num_beads_to_move):
        
        ind0, _, _, t3_trial, t2_trial, _ = one_bead_tangent_rotation_move(polymer, amp_move, num_beads)
        
        adjusted_beads.append(ind0)
        all_t3_trial.append(t3_trial)
        all_t2_trial.append(t2_trial)

    return adjusted_beads, all_t3_trial, all_t2_trial


all_moves = [MCAdapter(move) for move in [crank_shaft_move, end_pivot_move, slide_move]]

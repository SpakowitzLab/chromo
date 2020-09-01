"""
MCmove class

Creates a Monte Carlo move object that contains parameters for the particular
move.
"""

import numpy as np

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
        self.mcmove_on = True
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


def crank_shaft_move(polymer, amp_move, amp_bead):
    """Rotate section of polymer around axis formed by two bounding beads."""
    # Select ind0 and indf for the crank-shaft move
    delta_ind = min(np.random.randint(2, amp_bead), polymer.num_beads)
    ind0 = np.random.randint(polymer.num_beads - delta_ind + 1)
    indf = ind0 + delta_ind

    # Generate the rotation matrix and vector around the vector between bead ind0 and indf
    rot_angle = amp_move * (np.random.rand() - 0.5)

    if ind0 == (indf + 1):
        delta_t3 = polymer.t_3[ind0, :]
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
        t3_poly_trial[i_bead - ind0, :] = np.matmul(rot_matrix, polymer.t_3[i_bead, :])
    return ind0, indf, r_poly_trial, t3_poly_trial, None, None

all_moves = [MCAdapter(move) for move in [crank_shaft_move]]

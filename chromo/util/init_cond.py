"""
Setup the initial condition for the polymer chains.

The initial condition can be set using two options:

1. Random initialization
2. Initialization from file

"""
import numpy as np


def init_cond(length_bead, num_polymers, num_beads, num_epigenmark, from_file=False, input_dir="../input/"):
    """Create num_polymer of num_beads each either from_file or randomly."""
    num_total = num_polymers * num_beads  # Total number of beads

    length_bead_nm = length_bead * 0.34        # Length of bead (nm)

    if from_file:
        print("Initialize from saved conformation")
        r_poly = np.loadtxt(input_dir + "r_poly_0", delimiter=',')
        t3_poly = np.loadtxt(input_dir + "t3_poly_0", delimiter=',')
        epigen_bind = np.loadtxt(input_dir + "epigen_bind_0", delimiter=',')
    else:
        print("Initialize from random conformation")

        r_poly = np.zeros((num_total, 3), 'd')
        t3_poly = np.zeros((num_total, 3), 'd')
        t3_poly[:, 2] = 1.

        epigen_bind = np.zeros((num_total, num_epigenmark), 'd')

        for i_poly in range(num_polymers):
            ind0 = num_beads * i_poly

#            for i_bead in range(1, num_beads):
#                r_poly[ind0 + i_bead, :] = r_poly[i_bead-1, :] + length_bead_nm * t3_poly[i_bead-1, :]

        # Define the position vectors
        r_poly = r_poly + 0.5 * np.random.randn(num_total, 3)

    return r_poly, t3_poly, epigen_bind


def shift_vector(a, shift_index, num_beads, i_poly):
    """
    Generate a step forward/back vector by shift_index steps.

    input:  vector a        Full vector (length num_beads * num_polymers x 3)
            shift_index     Index to shift the vector
            num_beads       Number of beads in each polymer
            i_poly          Index of the polymer to output the shift vector

    output: a_shift         Shifted vector (length num_bead x 3)

    """
    ind0 = num_beads * i_poly  # Determine the zero index for i_poly
    # Shift over shift_index to be between 0 and num_beads
    shift_index = shift_index % num_beads

    mid_index = ind0 + shift_index
    end_index = ind0 + num_beads

    a_shift = np.concatenate([a[mid_index:end_index, :], a[ind0:mid_index, :]])

    return a_shift


def shift_array(a, shift_index, num_beads, i_poly):
    """
    Generate a step forward/back array by shift_index steps.

    input:  array a        Full array (length num_beads * num_polymers x 3)
            shift_index     Index to shift the vector
            num_beads       Number of beads in each polymer
            i_poly          Index of the polymer to output the shift vector

    output: a_shift         Shifted vector (length num_bead x 3)

    """
    ind0 = num_beads * i_poly  # Determine the zero index for i_poly
    # Shift over shift_index to be between 0 and num_beads
    shift_index = shift_index % num_beads

    mid_index = ind0 + shift_index
    end_index = ind0 + num_beads

    a_shift = np.concatenate([a[mid_index:end_index], a[ind0:mid_index]])

    return a_shift
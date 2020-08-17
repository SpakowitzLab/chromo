"""
Routines for performing Monte Carlo simulations.


"""

import numpy as np
from chromo.mc.calc_density import calc_density
from chromo.mc.mc_move import mc_move


def mc_sim(polymer, epigenmark, num_epigenmark, num_polymers, num_mc_steps, mcmove, num_mc_move_types, field):
    """
    Perform Monte Carlo simulation with the

    :param polymer:
    :param num_epigenmark:
    :param num_polymers:
    :param num_mc_steps:
    :param mcmove:
    :return:
    """

    # Re-evaluate the densities

    density = np.zeros((field.num_bins_total, num_epigenmark + 1), 'd')
    for i_poly in range(num_polymers):
        density_poly, index_xyz = calc_density(polymer[i_poly].r_poly, polymer[i_poly].epigen_bind,
                                      num_epigenmark, 0, polymer[i_poly].num_beads, field)
        density[index_xyz, :] += density_poly

    # Perform Monte Carlo simulation for num_mc_steps steps
    mc_count = 0
    while mc_count < num_mc_steps:

        for mc_move_type in range(num_mc_move_types):
            if mcmove[mc_move_type].move_on:
                for i_move_cycle in range(mcmove[mc_move_type].num_per_cycle):
                    mc_move(polymer, epigenmark, density, num_epigenmark, num_polymers, mcmove, mc_move_type, field)

        mc_count += 1

    return


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

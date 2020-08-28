"""Routines for performing Monte Carlo simulations."""

import numpy as np
from .calc_density import calc_density
from .mc_move import mc_move


def mc_sim(polymers, epigenmarks, num_mc_steps, mcmoves, field):
    """Perform Monte Carlo simulation."""
    num_polymers = len(polymers)
    num_epigenmark = len(epigenmarks)
    # Re-evaluate the densities
    density = np.zeros((field.num_bins_total, num_epigenmark + 1), 'd')
    for poly in polymers:
        density_poly, index_xyz = calc_density(
                poly.r, poly.states, 0, poly.num_beads, field
        )
        density[index_xyz, :] += density_poly

    # Perform Monte Carlo simulation for num_mc_steps steps
    mc_count = 0
    while mc_count < num_mc_steps:
        for mcmove in mcmoves:
            if mcmove.move_on:
                for i_move_cycle in range(mcmove.num_per_cycle):
                    mc_move(polymers, epigenmarks, density, num_epigenmark,
                            num_polymers, mcmove, field)

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

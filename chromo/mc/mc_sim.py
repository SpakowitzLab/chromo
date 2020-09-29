"""Routines for performing Monte Carlo simulations."""
import numpy as np


def mc_sim(polymers, epigenmarks, num_mc_steps, mc_moves, field):
    """Perform Monte Carlo simulation."""
    for i in range(num_mc_steps):

        if (i+1) % 500 == 0:
            print("MC Step " + str(i+1) + " of " + str(num_mc_steps))

        for adaptible_move in mc_moves:
            if adaptible_move.move_on:
                for j in range(adaptible_move.num_per_cycle):
                    for poly in polymers:
                        mc_step(adaptible_move, poly, epigenmarks, field)


def mc_step(adaptible_move, poly, epigenmarks, field):
    """Compute energy change and determine move acceptance."""
    # get proposed state
    proposal = adaptible_move.propose(poly)

    # compute change in energy
    dE = 0
    dE += poly.compute_dE(*proposal)
    if poly in field:
        dE += field.compute_dE(*proposal)

    # accept move
    if np.random.rand() < np.exp(-dE):
        adaptible_move.accept(poly, *proposal)

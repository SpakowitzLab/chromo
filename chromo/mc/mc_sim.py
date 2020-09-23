"""Routines for performing Monte Carlo simulations."""

import numpy as np

def mc_sim(polymers, epigenmarks, num_mc_steps, mc_moves, field):
    """Perform Monte Carlo simulation."""
    for i in range(num_mc_steps):
        for adaptible_move in mc_moves:
            if adaptible_move.move_on:
                for j in range(adaptible_move.num_per_cycle):
                    for poly in polymers:
                        mc_step(adaptible_move, poly, epigenmarks, field)
    return

def mc_step(adaptible_move, poly, epigenmarks, field):
    # get proposed state
    ind0, indf, r, t3, t2, states = adaptible_move.propose(poly)
    # compute change in energy
    dE = 0
    dE += poly.compute_dE(ind0, indf, r, t3, t2, states)
    if poly in field:
        dE += field.compute_dE(poly, ind0, indf, r, t3, t2, states)
    if np.random.rand() < np.exp(-dE):
        adaptible_move.accept()
        poly.r[ind0:indf, :] = r
        poly.t3[ind0:indf, :] = t3
        poly.t2[ind0:indf, :] = t2
        poly.states[ind0:indf, :] = states
    return

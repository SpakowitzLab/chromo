"""Routines for performing Monte Carlo simulations."""
from pathlib import Path

import numpy as np

import chromo.mc.adapt as adapt


def mc_sim(
    polymers, epigenmarks, num_mc_steps, mc_moves, field, adapter, output_dir):
    """Perform Monte Carlo simulation."""

    for adaptible_move in mc_moves:
        adaptible_move.bead_amp_range[1] = min(
            [poly.num_beads for poly in polymers]) - 1
    
    for i in range(num_mc_steps):
        
        if (i+1) % 500 == 0:
            print("MC Step " + str(i+1) + " of " + str(num_mc_steps))

        for adaptible_move in mc_moves:    
            if adaptible_move.move_on:
                for j in range(adaptible_move.num_per_cycle):
                    for poly in polymers:
                        mc_step(adaptible_move, poly, epigenmarks, field)

                # Adapt moves based on acceptance rate if startup complete
                if not adaptible_move.performance_tracker.startup:
                    adaptible_move = adapter(adaptible_move)
                else:
                    adaptible_move.performance_tracker.store_performance(
                        adaptible_move.amp_bead, adaptible_move.amp_move)

                # Output model performance
                adaptible_move.performance_tracker.save_performance(
                    output_dir/Path(f"{adaptible_move.name}-acceptance_log.csv"))
                    

def mc_step(adaptible_move, poly, epigenmarks, field):
    """Compute energy change and determine move acceptance."""
    
    # get proposed state
    proposal = adaptible_move.propose(poly)

    # compute change in energy
    dE = 0
    dE += poly.compute_dE(*proposal)
    if poly in field:
        dE += field.compute_dE(poly, *proposal)

    # accept move
    if np.random.rand() < np.exp(-dE):
        adaptible_move.accept(poly, *proposal)
    else:
        adaptible_move.reject()

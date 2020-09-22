"""Monte Carlo simulations of a discrete wormlike chain."""
from pathlib import Path

import numpy as np

from .mc_sim import mc_sim
from .moves import all_moves


def polymer_in_field(polymers, epigenmarks, field, num_mc_steps, num_save_mc,
                     mc_moves=None, output_dir='.'):
    """
    Monte Carlo simulation of a tssWLC in a field.

    This example code can be used to understand how to call the codebase, or
    run directly for simple simulations. See the documentation for the

    Parameters
    ----------
    polymers : Sequence[Polymer]
        The polymers to be simulated. Epigenetic marks, discretization, and
        other parameters should be specified before input.
    epigenmarks : Sequence[Epigenmark]
        DEPRECATED. Should be integrated into the "Polymer" class.
    field : Field
        The discretization of space in which to simulate the polymers.
    num_save_mc : int
        How many Monte Carlo steps to take between saving the simulation state.
    mc_moves : Optional[Sequence[int]]
        ID of each monte carlo move desired. Default of None uses all moves.
    output_dir : Optional[Path], default: '.'
        Directory in which to save the simulation output.
    """
    if mc_moves is None:
        mc_moves = all_moves
    # Perform Monte Carlo simulation for each save file
    for mc_count in range(num_save_mc):
        mc_sim(polymers, epigenmarks, num_mc_steps, mc_moves, field)
        for poly in polymers:
            poly.write_repr(output_dir / Path(f"{poly.name}-{mc_count}.npz"))
        print("Save point " + str(mc_count) + " completed")

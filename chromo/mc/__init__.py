"""
Monte Carlo simulations of a discrete wormlike chain.
"""
from pathlib import Path

import numpy as np

from .mc_sim import mc_sim
from ..MCmove import MCmove #TODO fix this redundancy    

def polymer_in_field(polymers, epigenmarks, field, mc_moves, num_mc_steps, num_save_mc, output_dir):
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
    mc_moves : Sequence[int]
        ID of each monte carlo move desired.
    num_mc_steps : int
        How many total Monte Carlo steps to take.
    num_save_mc : int
        How many Monte Carlo steps to take between saving the simulation state.
    output_dir : Optional[Path], default: '.'
        Directory in which to save the simulation output.
    seed : int
        How to initialize the MT19937 RNG. See
        https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html
        for best practices.
    """
    num_epigenmark = len(epigenmarks)
    num_polymers = len(polymers) #TODO remove this
    mc_moves = [MCmove(i) for i in mc_moves]
    num_mc_move_types = len(mc_moves) #TODO remove this as well
    # Perform Monte Carlo simulation for each save file
    for mc_count in range(1, num_save_mc + 1):
        mc_sim(polymers, epigenmarks, num_epigenmark, num_polymers, num_mc_steps, mc_moves, num_mc_move_types, field)

        save_file(polymers, num_polymers, mc_count, output_dir)
        print("Save point " + str(mc_count) + " completed")


def save_file(polymers, num_polymers, file_count, home_dir='.'):
    """
    Save the conformations to the output directory (output_dir).

    Saves the input *r_poly* to the file ``"r_poly_{file_count}"``, and
    saves each of the *ti_poly* to the file ``"t{i}_poly_{file_count}"``.

    Parameters
    ----------
    r_poly, t3_poly : (3, N) array_like
        The chain conformation information (to be saved).
    file_count : int
        A unique file index to append to the filename for this save point.
    home_dir : Path
    :param home_dir:
    :param file_count:
    :param polymers:
    :param num_polymers:
    """

    r_poly_total = polymers[0].r_poly
    t3_poly_total = polymers[0].t3_poly
    for i_poly in range(1, num_polymers):
        r_poly_total.append(polymers[i_poly].r_poly)
        t3_poly_total.append(polymers[i_poly].t3_poly)

    home_dir = Path(home_dir)
    conformations = {f"r_poly_{file_count}": r_poly_total,
                     f"t3_poly_{file_count}": t3_poly_total}
    for name, data in conformations.items():
        np.savetxt(home_dir / Path(name), data, delimiter=',')

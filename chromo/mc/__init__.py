"""Monte Carlo simulations of a discrete wormlike chain."""
from pathlib import Path
import warnings

import numpy as np

import chromo.mc.adapt as adapt
from .mc_sim import mc_sim
from .moves import all_moves
from ..util.reproducibility import make_reproducible
from ..components import Polymer
from ..marks import get_by_name, make_mark_collection
from ..fields import UniformDensityField


@make_reproducible
def simple_mc(num_polymers, num_beads, bead_length, num_marks, num_save_mc,
              num_saves, x_width, nx, y_width, ny, z_width, nz, random_seed=0,
              output_dir='.'):
    polymers = [
        Polymer.straight_line_in_x(
            f'Polymer-{i}', num_beads, bead_length,
            states=np.zeros((num_beads, num_marks)),
            mark_names=num_marks*['HP1']
        ) for i in range(num_polymers)
    ]
    marks = [get_by_name('HP1') for i in range(num_marks)]
    marks = make_mark_collection(marks)
    field = UniformDensityField(polymers, marks, x_width, nx, y_width, ny,
                                z_width, nz)
    return _polymer_in_field(polymers, marks, field, num_save_mc, num_saves,
                             random_seed=random_seed, output_dir=output_dir)


def _polymer_in_field(polymers, marks, field, num_save_mc, num_saves,
    adapter=adapt.feedback_adaption, mc_moves=None, random_seed=0, 
    output_dir='.'):
    """
    Monte Carlo simulation of a tssWLC in a field.

    This example code can be used to understand how to call the codebase, or
    run directly for simple simulations.

    Parameters
    ----------
    polymers : Sequence[Polymer]
        The polymers to be simulated.
    epigenmarks : `pd.DataFrame`
        Output of `chromo.marks.make_mark_collection`. Summarizes the energetic
        properties of each chemical modification.
    field : Field
        The discretization of space in which to simulate the polymers.
    num_save_mc : int
        How many Monte Carlo steps to take between saving the simulation state.
    adapter : adapter (optional, default `feedback_adaption`)
        Move adapter to adjust move and bead amplitudes in responce to MC move
        acceptance rates
    mc_moves : Optional[Sequence[int]]
        ID of each monte carlo move desired. Default of None uses all moves.
    output_dir : Optional[Path], default: '.'
        Directory in which to save the simulation output.
    """
    warnings.warn("The random seed is currently ignored.", UserWarning)
    if mc_moves is None:
        mc_moves = all_moves
    
    # Perform Monte Carlo simulation for each save file
    for mc_count in range(num_saves):
        mc_sim(
            polymers, marks, num_save_mc, mc_moves, field, adapter, output_dir)
        
        for poly in polymers:
            poly.to_csv(output_dir / Path(f"{poly.name}-{mc_count}.csv"))
        print("Save point " + str(mc_count) + " completed")


polymer_in_field = make_reproducible(_polymer_in_field)

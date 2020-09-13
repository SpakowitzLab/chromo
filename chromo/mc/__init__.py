"""Monte Carlo simulations of a discrete wormlike chain."""
from pathlib import Path
import warnings

from .mc_sim import mc_sim
from .moves import all_moves
from ..util.reproducibility import make_reproducible


@make_reproducible
def polymer_in_field(polymers, marks, field, num_save_mc, num_saves,
                     mc_moves=None, random_seed=0, output_dir='.'):
    """
    Monte Carlo simulation of a tssWLC in a field.

    This example code can be used to understand how to call the codebase, or
    run directly for simple simulations. See the documentation for the

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
        mc_sim(polymers, marks, num_save_mc, mc_moves, field)
        for poly in polymers:
            poly.to_csv(output_dir / Path(f"{poly.name}-{mc_count}.csv"))
        print("Save point " + str(mc_count) + " completed")

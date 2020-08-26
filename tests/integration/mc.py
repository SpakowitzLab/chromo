"""Tests that simply check if a simulation runs at all."""
import chromo.mc as mc
from chromo.Polymer import Polymer
from chromo.Field import Field

import numpy as np

from pathlib import Path
integration_dir = Path(__file__).parent.absolute()


def andy_mc():
    """
    Original simulation, written by Andy.

    Should be replaced with a cleaner version that doesn't rely on reading
    input files, except for optionally.
    """
    # Initialize the simulation by reading parameters from file (located in the
    # input directory)
    input_dir = integration_dir / Path("input")
    input_file = input_dir / Path("sim_input")
    num_polymers = np.genfromtxt(input_file, comments='#', dtype=int)[0]
    length_bead = np.genfromtxt(input_file, comments='#', dtype=float)[1]
    num_epigenmark = np.genfromtxt(input_file, comments='#', dtype=int)[2]
    length_box_x = np.genfromtxt(input_file, comments='#', dtype=float)[3]
    num_bins_x = np.genfromtxt(input_file, comments='#', dtype=int)[4]
    length_box_y = np.genfromtxt(input_file, comments='#', dtype=float)[5]
    num_bins_y = np.genfromtxt(input_file, comments='#', dtype=int)[6]
    length_box_z = np.genfromtxt(input_file, comments='#', dtype=float)[7]
    num_bins_z = np.genfromtxt(input_file, comments='#', dtype=int)[8]
    num_save_mc = np.genfromtxt(input_file, comments='#', dtype=int)[9]
    num_mc_steps = np.genfromtxt(input_file, comments='#', dtype=int)[10]
    # from_file = np.genfromtxt(input_file, comments='#', dtype=bool)[11]
    output_dir = np.genfromtxt(input_file, comments='#', dtype=str)[12]
    beads_per_polymer = np.genfromtxt(input_file, comments='#', dtype=str)[13]
    num_mc_move_types = 1

    # Create the epigenetic marks from the Epigenmark class if "epigen_prop"
    # files exist
    epigenmarks = []
    prop_file = input_dir / Path("epigen_prop")
    if not prop_file.exists():
        raise OSError(f"Property file does not exist for epigenetic marks")
    for epigenmark_count in range(1, num_epigenmark + 1):
        epigenmarks.append(Epigenmark(epigenmark_count, input_dir))

    # Create the chromosomal polymers from the Polymer class
    # 1. Define the sequence of epigenetic marks if "sequence" files exist for
    #    all epigenetic marks
    # 2. Initialize the conformations from file (if from_file == True) or from
    #    random initiatlization

    # Create the polymers from the Polymer class if "chromo_prop" files exist
    polymers = []
    for chromo_count in range(num_polymers):
        polymers.append(Polymer.straight_line_in_x(beads_per_polymer,
                                                   length_bead))

    # Setup the Monte Carlo class to define the properties of the simulation
    mc_moves = np.arange(num_mc_move_types)
    field = Field(length_box_x, num_bins_x, length_box_y, num_bins_y,
                  length_box_z, num_bins_z)

    return mc.polymer_in_field(polymers, epigenmarks, field, mc_moves,
                               num_mc_steps, num_save_mc, output_dir)

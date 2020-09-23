"""Tests that simply check if a simulation runs at all."""
import chromo.mc as mc
from chromo.components import *
from chromo.marks import *
from chromo.fields import *

import numpy as np
import pandas as pd

from pathlib import Path
integration_dir = Path(__file__).parent.absolute()


def andy_mc():
    """
    Original simulation, written by Andy.

    Should be replaced with a cleaner version that doesn't rely on reading
    input files, except for optionally.

    More importantly, should rewrite so that there's not a bunch of "num_"
    stuff that's redundant/maybe buggy.
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
    num_beads = np.genfromtxt(input_file, comments='#', dtype=int)[13]
    num_mc_move_types = 1

    # Create the epigenetic marks from the Epigenmark class if "epigen_prop"
    # files exist
    epigenmarks = []
    epi_files = list(input_dir.glob("epigen_prop*"))
    if len(epi_files) != num_epigenmark:
        raise ValueError("There should be one epigen_prop file per mark!")
    for epi_file in epi_files:
        epi_info = pd.read_csv(epi_file, comment='#',
                               delim_whitespace=True).iloc[:, 0]
        epigenmarks.append(Epigenmark(epi_info.name, *epi_info.values))

    seq_files = list(input_dir.glob("seq*"))
    if len(seq_files) != num_epigenmark:
        raise ValueError("There should be one seq file per mark!")
    states = np.zeros((num_beads, num_epigenmark))
    for i, seq_file in enumerate(seq_files):
        states[:, i] = np.genfromtxt(seq_file, comments='#', dtype=int)
    # Create the polymers from the Polymer class if "chromo_prop" files exist
    polymers = []
    bead_counts = pd.read_csv(input_dir / Path('chromo_prop'),
                              delim_whitespace=True)
    if len(bead_counts) != num_polymers:
        raise ValueError("There should be one chrom_prop entry per polymer!")
    for _, (name, bead_count) in bead_counts.iterrows():
        if bead_count != num_beads:
            raise ValueError("There are two redundant ways to specify "
                             "num_beads, and you gave two different values.")
        polymers.append(Polymer.straight_line_in_x(
                name, epigenmarks, states, bead_count, length_bead))

    # Setup the Monte Carlo class to define the properties of the simulation
    field = UniformDensityField(polymers, length_box_x, num_bins_x, length_box_y,
                                num_bins_y, length_box_z, num_bins_z)

    return mc.polymer_in_field(polymers, epigenmarks, field, num_mc_steps,
                               num_save_mc, None, output_dir)

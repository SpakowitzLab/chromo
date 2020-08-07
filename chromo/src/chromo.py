"""
Monte Carlo simulations of a discrete wormlike chain.

"""
from pathlib import Path

import numpy as np
from util.Epigenmark import Epigenmark
from util.Polymer import Polymer
from util.MCmove import MCmove
from util.Field import Field
import os.path
from mc.mc_sim import mc_sim

def main():
    """Show example of MC simulation code usage."""

    # Initialize the simulation by reading parameters from file (located in the input directory)
    input_dir = "../input/"
    num_polymers = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[0]
    length_bead = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[1]
    num_epigenmark = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[2]
    length_box_x = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[3]
    num_bins_x = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[4]
    length_box_y = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[5]
    num_bins_y = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[6]
    length_box_z = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=float)[7]
    num_bins_z = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[8]
    num_save_mc = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[9]
    num_mc_steps = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=int)[10]
    from_file = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=bool)[11]
    output_dir = np.genfromtxt(input_dir + "sim_input", comments='#', dtype=str)[12]
    num_mc_move_types = 1

    # Create the epigenetic marks from the Epigenmark class if "epigen_prop" files exist
    epigenmark = []
    prop_file = input_dir + "epigen_prop"
    if os.path.isfile(prop_file):
        for epigenmark_count in range(1, num_epigenmark + 1):
            epigenmark.append(Epigenmark(epigenmark_count, input_dir))
    else:
        print(f"Property file does not exist for epigenetic marks")
        exit()

    # Create the chromosomal polymers from the Polymer class
    # 1. Define the sequence of epigenetic marks if "sequence" files exist for all epigenetic marks
    # 2. Initialize the conformations from file (if from_file == True) or from random initiatlization

    # Create the polymers from the Polymer class if "chromo_prop" files exist
    polymer = []
    prop_file = input_dir + "chromo_prop"
    if os.path.isfile(prop_file):
        for chromo_count in range(1, num_polymers + 1):
            polymer.append(Polymer(epigenmark, num_epigenmark, chromo_count, length_bead, from_file, input_dir))
    else:
        print(f"Property file does not exist for chromosomes")
        exit()

    # Setup the Monte Carlo class to define the properties of the simulation

    mcmove = []
    for mcmove_count in range(1, num_mc_move_types + 1):
        mcmove.append(MCmove(mcmove_count))
    field = Field(length_box_x, num_bins_x, length_box_y, num_bins_y, length_box_z, num_bins_z)

    # Perform Monte Carlo simulation for each save file

    for mc_count in range(1, num_save_mc + 1):
        mc_sim(polymer, epigenmark, num_epigenmark, num_polymers, num_mc_steps, mcmove, num_mc_move_types, field)

        save_file(polymer, num_polymers, mc_count, output_dir)
        print("Save point " + str(mc_count) + " completed")


def save_file(polymer, num_polymers, file_count, home_dir='.'):
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
    :param polymer:
    :param num_polymers:
    """

    r_poly_total = polymer[0].r_poly
    t3_poly_total = polymer[0].t3_poly
    for i_poly in range(1, num_polymers):
        r_poly_total.append(polymer[i_poly].r_poly)
        t3_poly_total.append(polymer[i_poly].t3_poly)

    home_dir = Path(home_dir)
    conformations = {f"r_poly_{file_count}": r_poly_total,
                     f"t3_poly_{file_count}": t3_poly_total}
    for name, data in conformations.items():
        np.savetxt(home_dir / Path(name), data, delimiter=',')


if __name__ == "__main__":
    main()

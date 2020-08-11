"""
Polymer class

Creates a polymer with a defined length and discretization

"""

import numpy as np
from util.init_cond import init_cond
from util.find_parameters import find_parameters
import os.path


class Polymer:
    type = "Chromosomal polymer"

    def __init__(self, epigenmark, num_epigenmark, polymer_count, length_bead, from_file, input_dir):
        # Initialize the properties of the polymers
        ind0 = 2 * (polymer_count - 1)
        self.polymer_count = polymer_count
        self.input_dir = input_dir
        self.name = np.genfromtxt(input_dir + "chromo_prop", comments='#', dtype=str)[ind0]
        self.num_beads = np.genfromtxt(input_dir + "chromo_prop", comments='#', dtype=int)[ind0 + 1]
        self.num_nucleo_per_bead = 1
        self.max_bound = self.num_nucleo_per_bead * 2

        # Initialize the conformation, protein binding state, and the sequence of epigenetic marks
        self.r_poly, self.t3_poly, self.epigen_bind = init_cond(length_bead, 1, self.num_beads,
                                                                num_epigenmark, from_file, self.input_dir)

        # Load the polymer parameters for the conformational energy
        self.sim_type, self.eps_bend, self.eps_par, self.eps_perp, self.gamma, self.eta = find_parameters(length_bead)

        self.sequence = np.zeros((self.num_beads, num_epigenmark), 'd')
        for epigenmark_count in range(1, num_epigenmark + 1):
            seq_file = self.input_dir + "chromo" + str(self.polymer_count) + "seq" + str(epigenmark_count)
            if os.path.isfile(seq_file):
                self.sequence[:, epigenmark_count - 1] = np.loadtxt(seq_file, delimiter=',')
            else:
                print(f"Sequence file does not exist for chromosome {self.polymer_count}")
                exit()

    def __str__(self):
        return f"{self.name} is a polymer with {self.num_beads} beads (each bead with " \
               f"{self.num_nucleo_per_bead} nucleosomes)"




"""
Epigenetic mark class

Creates an epigenetic mark with a defined sequence and properties

"""

import numpy as np


class Epigenmark:
    type = "Epigenetic Mark"

    def __init__(self, epimark_count, input_dir):
        ind0 = 5 * (epimark_count - 1)
        self.input_dir = input_dir
        self.name = np.genfromtxt(input_dir + "epigen_prop", comments='#', dtype=str)[ind0 + 1]
        self.bind_energy = np.genfromtxt(input_dir + "epigen_prop", comments='#', dtype=float)[ind0 + 2]
        self.int_energy = np.genfromtxt(input_dir + "epigen_prop", comments='#', dtype=float)[ind0 + 3]
        self.chem_pot = np.genfromtxt(input_dir + "epigen_prop", comments='#', dtype=float)[ind0 + 4]

    def __str__(self):
        return f"{self.name} is an epigenetic mark. " \
               f"Binding energy {self.bind_energy} and interaction energy {self.int_energy}"



"""Run a basic simulation from a script.

This simulation runs a Monte Carlo simulation from a script for easier use of
a remote tower.
"""

# Built-in Modules
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

# External Modules
import numpy as np

# Custom Modules
import chromo
import chromo.mc as mc
from chromo.polymers import Chromatin
import chromo.marks
from chromo.fields import UniformDensityField

# Change working directory
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print("System path: ")
print(sys.path)

# Specify epigenetic mark
epimarks = [
    chromo.marks.get_by_name('HP1'),

    # For now, we will treat PRC1 as having the same properties as HP1
    chromo.marks.get_by_name('PRC1')
]

chemical_potential_HP1 = float(sys.argv[1])
epimarks[0].chemical_potential = chemical_potential_HP1

chemical_potential_Polycomb = float(sys.argv[2])
epimarks[1].chemical_potential = chemical_potential_Polycomb

HP1_PRC1_cross_talk_interaction_energy = float(sys.argv[3])
epimarks[0].cross_talk_interaction_energy["PRC1"] = \
    HP1_PRC1_cross_talk_interaction_energy

print("Epigenetic marks: ")
print(epimarks)

# Reformat epigenic marks into a dataframe format
marks = chromo.marks.make_mark_collection(
    epimarks
)

# Confine to spherical chrom. territory 1800 um diameter (Cremer & Cremer 2001)
confine_type = "Spherical"
confine_length = 900

print("Constructing polymer...")
# Specify polymers length
num_beads = 25000
bead_spacing = 16.5        # 50-bp linker length
chem_mods_path = np.array(
    [
        "chromo/chemical_mods/HNCFF683HCZ_H3K9me3_methyl_25000.txt",
        "chromo/chemical_mods/ENCFF919DOR_H3K27me3_methyl_25000.txt"
    ]
)
chemical_mods = Chromatin.load_seqs(chem_mods_path)

p = Chromatin.confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_spacing,
    states=chemical_mods.copy(),
    confine_type=confine_type,
    confine_length=confine_length,
    mark_names=np.array(['HP1', 'PRC1']),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array(['H3K9me3', 'H3K27me3'])
)

# Specify the field containing the polymers
n_bins_x = 63
x_width = 2 * confine_length
n_bins_y = n_bins_x
y_width = x_width
n_bins_z = n_bins_x
z_width = x_width
udf = UniformDensityField(
    [p], marks, x_width, n_bins_x, y_width,
    n_bins_y, z_width, n_bins_z, confine_type=confine_type,
    confine_length=confine_length
)

# Specify the bead selection and move amplitude bounds
polymers = [p]
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers)

if __name__ == "__main__":
    """Run the simulation.
    """
    print("Starting new simulation...")
    num_snapshots = 200
    mc_steps_per_snapshot = 500000
    mc.polymer_in_field(
        [p],
        marks,
        udf,
        mc_steps_per_snapshot,
        num_snapshots,
        amp_bead_bounds,
        amp_move_bounds,
        output_dir='output'
    )
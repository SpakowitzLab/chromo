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
import chromo.binders
from chromo.fields import UniformDensityField

# Change working directory
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print("System path: ")
print(sys.path)

# Specify reader proteins
binders = [
    chromo.binders.get_by_name('HP1'),

    # For now, we will treat PRC1 as having the same properties as HP1
    chromo.binders.get_by_name('PRC1')
]

chemical_potential_HP1 = -0.4
binders[0].chemical_potential = chemical_potential_HP1

chemical_potential_Polycomb = -0.4
binders[1].chemical_potential = chemical_potential_Polycomb

HP1_PRC1_cross_talk_interaction_energy = -2.0
binders[0].cross_talk_interaction_energy["PRC1"] = \
    HP1_PRC1_cross_talk_interaction_energy

print("Reader Proteins: ")
print(binders)

# Reformat reader proteins into a dataframe format
binders = chromo.binders.make_binder_collection(binders)

# Confine to spherical chrom. territory 1800 um diameter (Cremer & Cremer 2001)
confine_type = "Spherical"
confine_length = 900

print("Constructing polymer...")
# Specify polymers length
num_beads = int(sys.argv[1])    # 393216
bead_spacing = 16.5             # 50-bp linker length

# Scale down the confinement so the density matches that of a chromosome
# inside a chromosome territory
frac_full_chromo = num_beads / 393216
confine_length *= np.cbrt(frac_full_chromo)

chem_mods_path = np.array(
    [
        "chromo/chemical_mods/HNCFF683HCZ_H3K9me3_methyl.txt",
        "chromo/chemical_mods/ENCFF919DOR_H3K27me3_methyl.txt"
    ]
)
chemical_mods_all = Chromatin.load_seqs(chem_mods_path)


def run_sim(seed):
    """Run a simulation
    """

    start_ind = np.random.randint(chemical_mods_all.shape[0])
    chemical_mods = np.take(
        chemical_mods_all,
        np.arange(start_ind, start_ind + num_beads),
        mode="wrap",
        axis=0
    )

    p = Chromatin.confined_gaussian_walk(
        'Chr-1',
        num_beads,
        bead_spacing,
        states=np.zeros(chemical_mods.shape, dtype=int),
        confine_type=confine_type,
        confine_length=confine_length,
        binder_names=np.array(['HP1', 'PRC1']),
        chemical_mods=chemical_mods,
        chemical_mod_names=np.array(['H3K9me3', 'H3K27me3'])
    )

    # Specify the field containing the polymers
    n_bins_x = int(round(63 * np.cbrt(frac_full_chromo)))
    x_width = 2 * confine_length
    n_bins_y = n_bins_x
    y_width = x_width
    n_bins_z = n_bins_x
    z_width = x_width
    udf = UniformDensityField(
        [p], binders, x_width, n_bins_x, y_width,
        n_bins_y, z_width, n_bins_z, confine_type=confine_type,
        confine_length=confine_length, chi=1, fast_field=1
    )

    # Specify the bead selection and move amplitude bounds
    polymers = [p]
    amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers)

    print("Starting new simulation...")
    num_snapshots = 400
    mc_steps_per_snapshot = 50000
    mc.polymer_in_field(
        [p],
        binders,
        udf,
        mc_steps_per_snapshot,
        num_snapshots,
        amp_bead_bounds,
        amp_move_bounds,
        output_dir='output',
        random_seed=seed
    )


seeds = np.random.randint(0, 1E5, 5)
run_sim(seeds[0])

"""Run a basic simulation from a script.

This simulation runs a Monte Carlo simulation from a script for easier use of
a remote tower.
"""

# Built-in Modules
import os
import sys
from inspect import getmembers, isfunction

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

# External Modules
import numpy as np

# Custom Modules
import chromo.mc as mc
from chromo.polymers import Chromatin
import chromo.binders
from chromo.fields import UniformDensityField
import doc.tools.mu_schedules as ms

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

chemical_potential = -0.4
binders[0].chemical_potential = chemical_potential
binders[1].chemical_potential = chemical_potential

self_interaction_HP1 = float(sys.argv[1])
binders[0].interaction_energy = self_interaction_HP1

self_interaction_Polycomb = float(sys.argv[2])
binders[1].interaction_energy = self_interaction_Polycomb

HP1_PRC1_cross_talk_interaction_energy = float(sys.argv[3])
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
num_beads = 5000
bead_spacing = 16.5        # 50-bp linker length

# Scale down the confinement so the density matches that of a chromosome
# inside a chromosome territory
frac_full_chromo = num_beads / 393217
confine_length *= np.cbrt(frac_full_chromo)

# Define pattern of epigenetic modifications
chem_mods_path = np.array(
    [
        "chromo/chemical_mods/HNCFF683HCZ_H3K9me3_methyl_5000.txt",
        "chromo/chemical_mods/ENCFF919DOR_H3K27me3_methyl_5000.txt"
    ]
)
chemical_mods = Chromatin.load_seqs(chem_mods_path)

# Create a list of mu schedules, which will be defined in another file.
schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "linear_2_for_negative_cp"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

p = Chromatin.confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_spacing,
    states=chemical_mods.copy(),
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
    confine_length=confine_length, chi=1
)

# Specify the bead selection and move amplitude bounds
polymers = [p]
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers)
seed = np.random.randint(0, 1E5)
mu_schedule = mu_schedules[0]

if __name__ == "__main__":
    """Run the simulation.
    """
    print("Starting new simulation...")
    num_snapshots = 200
    mc_steps_per_snapshot = 20000
    mc.polymer_in_field(
        [p],
        binders,
        udf,
        mc_steps_per_snapshot,
        num_snapshots,
        amp_bead_bounds,
        amp_move_bounds,
        output_dir='output',
        mu_schedule=mu_schedule,
        random_seed=seed
    )

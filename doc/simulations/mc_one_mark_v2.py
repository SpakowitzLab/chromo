"""Run a basic simulation from a script.

This simulation runs a Monte Carlo simulation from a script for easier use of
a remote tower.
"""

# Built-in Modules
import os
import sys
from multiprocessing import Pool
from inspect import getmembers, isfunction

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
import doc.tools.mu_schedules as ms

# Change working directory
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print("System path: ")
print(sys.path)

# Confine to spherical chrom. territory 1800 um diameter (Cremer & Cremer 2001)
confine_type = "Spherical"
confine_length = 900

print("Constructing polymer...")
# Specify polymers length
num_beads = 25000          # 393217
bead_spacing = 16.5        # About 50 bp linker length

# Scale down the confinement so the density matches that of a chromosome
# inside a chromosome territory
frac_full_chromo = num_beads / 393217
confine_length *= np.cbrt(frac_full_chromo)

chem_mods_path = np.array(
    ["chromo/chemical_mods/meth"]
)

chemical_mods_all = Chromatin.load_seqs(chem_mods_path)

""" Create a list of mu schedules, which will be defined in another file.

schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "tanh_1"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]
"""


def run_sim(args):
    """Run a simulation
    """
    # Specify reader proteins
    binders = [chromo.binders.get_by_name('HP1')]

    chemical_potential = args[0]
    binders[0].chemical_potential = chemical_potential

    print("Reader Proteins: ")
    print(binders)

    # Reformat reader proteins into a dataframe format
    binders = chromo.binders.make_binder_collection(binders)

    start_ind=0
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
        binder_names=np.array(['HP1']),
        chemical_mods=chemical_mods,
        chemical_mod_names=np.array(['H3K9me3'])
    )

    # Specify the field containing the polymers
    n_buffer = 2
    n_accessible = int(round(63 * np.cbrt(frac_full_chromo)))
    n_bins_x = n_accessible + n_buffer
    x_width = 2 * confine_length * (1 + n_buffer / n_accessible)
    n_bins_y = n_bins_x
    y_width = x_width
    n_bins_z = n_bins_x
    z_width = x_width
    udf = UniformDensityField(
        [p], binders, x_width, n_bins_x, y_width,
        n_bins_y, z_width, n_bins_z, confine_type=confine_type,
        confine_length=confine_length, chi=1, assume_fully_accessible=1,
        fast_field=1
    )

    # Specify the bead selection and move amplitude bounds
    polymers = [p]
    amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers)

    print("Starting new simulation...")
    num_snapshots = 300
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
        random_seed=args[1],
        # mu_schedule=mu_schedules[0],
    )


chemical_potentials = np.array([
    # -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4
    -4.0, -1.5, -0.4, 0.0, 1.0, 2.0, 3.0, 4.0
])
seeds = np.random.randint(0, 1E5, len(chemical_potentials))
args = [(chemical_potentials[i], seeds[i]) for i in range(len(seeds))]
pool = Pool(12)
pool.map(run_sim, args)

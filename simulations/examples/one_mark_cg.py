"""Full chromosome simulation with one mark and dynamic coarse-graining.

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 13, 2022

Usage:      `python one_mark_cg.py <BINDER_NAME> <MODIFICATION_NAME> <MODIFICATION_PATH> <CP_BINDER> <J_BINDER1> <CG_FACTOR> <OPTIONAL_RANDOM_SEED>`
"""

import os
import sys
from inspect import getmembers, isfunction

import numpy as np

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)
os.chdir(parent_dir)

print("Root Directory: ")
print(os.getcwd())

import chromo.mc as mc
from chromo.polymers import Chromatin
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.util.rediscretize as rd
import chromo.util.mu_schedules as ms

# Set the random seed
if len(sys.argv) == 8:
    random_seed = int(sys.argv[7])
else:
    random_seed = np.random.randint(0, 1E5)
np.random.seed(random_seed)

# Store details on simulation
path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"
root_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])

# Binders
binder_name = sys.argv[1]
modification_name = sys.argv[2]
modification_sequence = sys.argv[3]
binder = chromo.binders.get_by_name(binder_name)
binder.chemical_potential = float(sys.argv[4])
binder.interaction_energy = float(sys.argv[5])
binders = chromo.binders.make_binder_collection([binder])

# Confinement
confine_type = "Spherical"
confine_length = 900

# Polymer
num_beads = 393216
bead_spacing = 16.5
chem_mods_path = np.array([modification_sequence])
chem_mod_paths_abs = [f"{root_dir}/{path}" for path in chem_mods_path]
chemical_mods = Chromatin.load_seqs(chem_mods_path)[:num_beads]
states = np.zeros(chemical_mods.shape, dtype=int)
p = Chromatin.confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_length=bead_spacing,
    states=np.zeros(chemical_mods.shape, dtype=int),
    confine_type=confine_type,
    confine_length=confine_length,
    binder_names=np.array([binder_name]),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array([modification_name])
)

# Field
n_accessible = 63
n_buffer = 2
n_bins_x = n_accessible + n_buffer
x_width = 2 * confine_length * (1 + n_buffer/n_accessible)
n_bins_y = n_bins_x
y_width = x_width
n_bins_z = n_bins_x
z_width = x_width
udf = UniformDensityField(
    [p], binders, x_width, n_bins_x, y_width,
    n_bins_y, z_width, n_bins_z, confine_type=confine_type,
    confine_length=confine_length, chi=1,
    assume_fully_accessible=1, fast_field=1, n_points=1000
)

# Coarse-grain the polymer, field, and binders
cg_factor = int(sys.argv[6])
p_cg = rd.get_cg_chromatin(
    polymer=p,
    cg_factor=cg_factor,
    name_cg="Chr_CG"
)
udf_cg = rd.get_cg_udf(
    udf_refined_dict=udf.dict_,
    binders_refined=binders,
    cg_factor=cg_factor,
    polymers_cg=[p_cg]
)
binders_cg = rd.get_cg_binders(
    binders_refined=binders,
    cg_factor=cg_factor
)

# Specify simulation hyperparameters
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p_cg])
num_snapshots = 200
mc_steps_per_snapshot = 3000

# Create a list of mu schedules, which are defined in another file
schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "linear_step_for_negative_cp"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

# Run the simulation
polymers_cg = mc.polymer_in_field(
    [p_cg],
    binders_cg,
    udf_cg,
    mc_steps_per_snapshot,
    num_snapshots,
    amp_bead_bounds,
    amp_move_bounds,
    output_dir='output',
    mu_schedule=mu_schedules[0],
    random_seed=random_seed,
    path_to_run_script=path_to_run_script,
    path_to_chem_mods=chem_mod_paths_abs,
    run_command=run_command
)

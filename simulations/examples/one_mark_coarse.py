"""Run a coarse grained simulation with one epigenetic mark.

By:         Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       14 September 2023

Usage:  python one_mark_coarse.py <CONFIG_FILE_PATH>

where <CONFIG_FILE_PATH> is a path to a configuration file, specifying the
simulation parameters. See example configuration files for examples.
"""

import os
import sys
import json

cwd = os.getcwd()
parent_dir = os.path.join(cwd, "..", "..")
sys.path.insert(1, parent_dir)
os.chdir(parent_dir)

print("Root Directory: ")
print(os.getcwd())

from inspect import getmembers, isfunction
import numpy as np

import chromo.mc as mc
from chromo.polymers import Chromatin
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.util.rediscretize as rd
import chromo.util.mu_schedules as ms

# Load simulation parameters from config file
config_file_path = sys.argv[1]
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)
required_keys = [
    "binder_name",
    "modification_name",
    "modification_sequence_path",
    "chemical_potential",
    "interaction_energy",
]
for key in required_keys:
    assert key in config.keys(), f"Missing required key: {key}"

# Set the random seed
if "random_seed" in config:
    random_seed = config["random_seed"]
else:
    random_seed = np.random.randint(0, 1E5)
np.random.seed(random_seed)

# Store details on simulation
path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"
root_dir = "/".join(path_to_run_script.split("/")[:-3])
if "output_dir" in config:
    output_dir = config["output_dir"]  # given relative to root_dir
else:
    output_dir = "output"

# Binders
binder_name = config["binder_name"]
binder = chromo.binders.get_by_name(binder_name)
binder.chemical_potential = config["chemical_potential"]
binder.interaction_energy = config["interaction_energy"]
binders = chromo.binders.make_binder_collection([binder])

# Chemical Modifications
modification_name = config["modification_name"]
mod_seq_path = config["modification_sequence_path"]  # given as an absolute path
chem_mods_path = np.array([mod_seq_path])
chemical_mods = Chromatin.load_seqs(chem_mods_path)

# Bead density is defined by MacPherson et al. 2018
bead_density = 393216 / (4 / 3 * np.pi * 900 ** 3)

# Number of Beads
if "num_beads" in config:
    num_beads = config["num_beads"]
    chemical_mods = chemical_mods[:num_beads]
else:
    num_beads = len(chemical_mods)

# Binding states
states = np.zeros(chemical_mods.shape, dtype=int)

# Confinement
confine_type = "Spherical"
confine_length = (num_beads / bead_density / (4 / 3 * np.pi)) ** (1 / 3)

# Polymer
if "bead_spacing" in config:
    bead_spacing = config["bead_spacing"]
else:
    # Bead spacing is defined by MacPherson et al. 2018
    bead_spacing = 16.5
p = Chromatin.confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_length=bead_spacing,
    states=states,
    confine_type=confine_type,
    confine_length=confine_length,
    binder_names=np.array([binder_name]),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array([modification_name])
)

# Field
# Voxel dimension is defined by MacPherson et al. 2018
n_accessible = int(np.round((63 * confine_length) / 900))
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
    assume_fully_accessible=1, fast_field=0
)

# Coarse-grain the polymer, field, and binders
if "cg_factor" in config:
    cg_factor = config["cg_factor"]
else:
    cg_factor = 16
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
mc_steps_per_snapshot = 6000

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
    output_dir=output_dir,
    mu_schedule=mu_schedules[0],
    random_seed=random_seed,
    path_to_run_script=path_to_run_script,
    path_to_chem_mods=chem_mods_path,
    run_command=run_command,
    config_file_path=config_file_path
)

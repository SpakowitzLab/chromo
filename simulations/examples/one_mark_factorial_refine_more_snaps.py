"""Run full chromosome simulation initialized to a coarse-grained configuration.

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 13, 2022

Usage:      `python two_mark_factorial_refined_step.py <SIM_ID>`

...where `<SIM_ID>` denotes the integer simulation index of the coarse-grained
polymer model.
"""

import os
import sys
from inspect import getmembers, isfunction

import numpy as np
import pandas as pd

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
if len(sys.argv) == 3:
    random_seed = int(sys.argv[2])
else:
    random_seed = np.random.randint(0, 1E5)
np.random.seed(random_seed)

# Store details on simulation
path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"
root_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])

# Load coarse-grained simulation
sim_id = int(sys.argv[1])
polymer_prefix = "Chr"
output_dir = f"/scratch/users/jwakim/chromo_multi_lower_cp/output/sim_{sim_id}"
binder_path = f"{output_dir}/binders"
udf_path = f"{output_dir}/UniformDensityField"

# Specify confinement
confine_type = "Spherical"
confine_length = 900

# Specify simulation
num_beads = 393216
bead_spacing = 16.5
num_snapshots = 200
mc_steps_per_snapshot = 5000

# Load previous run arguments
with open(f"{output_dir}/sim_call.txt", "r") as f:
    command = f.readline()
prev_run_args = command.split(" ")
binder_name = prev_run_args[2]
modification_sequence = prev_run_args[4]
chem_mods_path = np.array([modification_sequence])
chem_mod_paths_abs = [f"{root_dir}/{path}" for path in chem_mods_path]
chemical_mods = Chromatin.load_seqs(chem_mods_path)[:num_beads]

# Load latest snapshot from coarse-grained simulation
files = os.listdir(output_dir)
files = [
    file for file in files
    if file.endswith(".csv") and file.startswith(polymer_prefix)
]
snaps = [int(file.split(".")[0].split("-")[-1]) for file in files]
files = [file for _, file in sorted(zip(snaps, files))]
latest_snap = files[-1]
latest_snap_path = f"{output_dir}/{latest_snap}"

# Create binders
binder = chromo.binders.get_by_name(binder_name)
binders_list = [binder]
df_binders = pd.read_csv(binder_path, index_col="name")
cp_binder = df_binders.loc[binder_name, "chemical_potential"]
self_interact_binder = df_binders.loc[binder_name, "interaction_energy"]
binders_list[0].chemical_potential = float(cp_binder)
binders_list[0].interaction_energy = float(self_interact_binder)
binders = chromo.binders.make_binder_collection(binders_list)

# Create coarse-grained polymer
p = Chromatin.from_file(latest_snap_path, name="Chr_refine")

# Load field parameters from coarse-grained simulation
field_params = pd.read_csv(
    udf_path, header=None, names=["Attribute", "Value"], index_col=0
)
x_width = float(field_params.loc["x_width", "Value"])
y_width = float(field_params.loc["y_width", "Value"])
z_width = float(field_params.loc["z_width", "Value"])
nx = int(field_params.loc["nx", "Value"])
ny = int(field_params.loc["ny", "Value"])
nz = int(field_params.loc["nz", "Value"])
confine_type_cg = field_params.loc["confine_type", "Value"]
confine_length_cg = float(field_params.loc["confine_length", "Value"])
chi = float(field_params.loc["chi", "Value"])
assume_fully_accessible = (
    field_params.loc["assume_fully_accessible", "Value"] == "True"
)
fast_field = int(field_params.loc["fast_field", "Value"] == "True")

# Create coarse-grained field
udf = UniformDensityField(
    [p], binders, x_width, nx, y_width, ny, z_width, nz,
    confine_type=confine_type_cg, confine_length=confine_length_cg,
    chi=chi, assume_fully_accessible=assume_fully_accessible,
    fast_field=fast_field
)

# Specify move and bead amplitudes
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p])

# Run the refined simulation
polymers_refined = mc.polymer_in_field(
    [p],
    binders,
    udf,
    mc_steps_per_snapshot,
    num_snapshots,
    amp_bead_bounds,
    amp_move_bounds,
    output_dir='output_more_steps',
    random_seed=random_seed,
    path_to_run_script=path_to_run_script,
    path_to_chem_mods=chem_mod_paths_abs,
    run_command=run_command
)

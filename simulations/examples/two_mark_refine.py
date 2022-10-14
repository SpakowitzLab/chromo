"""Run full chromosome simulation initialized to a coarse-grained configuration.

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 13, 2022

Usage:      `python two_mark_refine.py <SIM_OUT_PATH> <SIM_ID> <REFINE_FACTOR> <OPTIONAL_RANDOM_SEED>`

...where `<SIM_OUT_PATH>` is the path to the output directory of the simulation,
`<SIM_ID>` denotes the integer simulation index of the coarse-grained polymer
model, `<REFINE_FACTOR>` denotes the factor increase in beads in the refined
model, and <OPTIONAL_RANDOM_SEED> provides an optional random seed.

The objective of this file is to enable multiple stages of simulation with
progressively increasing levels of refinement. Before this file, the simulator
was designed to accomodate two stages: a single coarse-grained stage and a
single refined stage. This setup provides more flexibility.
"""

import os
import sys
from inspect import getmembers, isfunction
import json

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
if len(sys.argv) == 5:
    random_seed = int(sys.argv[4])
else:
    random_seed = np.random.randint(0, 1E5)
np.random.seed(random_seed)

# Store details on simulation
path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"
root_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])

# Load coarse-grained simulation
sim_out_path = sys.argv[1]
sim_id = int(sys.argv[2])
polymer_prefix = "Chr"
output_dir = f"{sim_out_path}/sim_{sim_id}"
binder_path = f"{output_dir}/binders"
udf_path = f"{output_dir}/UniformDensityField"

# Specify confinement
confine_type = "Spherical"
confine_length = 900

# Specify simulation
true_num_beads = 393216
bead_spacing = 16.5
num_snapshots = 200
mc_steps_per_snapshot = 5000

# Load the latest snapshot from coarse-grained simulation
files = os.listdir(output_dir)
files = [
    file for file in files
    if file.endswith(".csv") and file.startswith(polymer_prefix)
]
snaps = [int(file.split(".")[0].split("-")[-1]) for file in files]
files = [file for _, file in sorted(zip(snaps, files))]
latest_snap = files[-1]
latest_snap_path = f"{output_dir}/{latest_snap}"

# Create coarse-grained polymer
p_cg = Chromatin.from_file(latest_snap_path, name="Chr_CG")
num_beads_cg = p_cg.num_beads
refine_factor = float(sys.argv[3])
new_cg_factor = int(np.floor(true_num_beads / num_beads_cg) / refine_factor)
num_beads = true_num_beads / new_cg_factor

# Specify chemical modifications
chem_mods_path = np.array([
    "chromo/chemical_mods/HNCFF683HCZ_H3K9me3_methyl.txt",
    "chromo/chemical_mods/ENCFF919DOR_H3K27me3_methyl.txt"
])
chem_mod_paths_abs = [f"{root_dir}/{path}" for path in chem_mods_path]
true_chemical_mods = Chromatin.load_seqs(chem_mods_path)[:true_num_beads]

intervals = rd.get_cg_bead_intervals(true_num_beads, cg_factor=new_cg_factor)
chemical_mods = rd.get_majority_state_in_interval(true_chemical_mods, intervals)

# Create binders
hp1 = chromo.binders.get_by_name("HP1")
prc1 = chromo.binders.get_by_name("PRC1")
binders_list = [hp1, prc1]
df_binders = pd.read_csv(binder_path, index_col="name")
cp_HP1 = df_binders.loc["HP1", "chemical_potential"]
cp_PRC1 = df_binders.loc["PRC1", "chemical_potential"]
self_interact_HP1 = df_binders.loc["HP1", "interaction_energy"]
self_interact_PRC1 = df_binders.loc["PRC1", "interaction_energy"]
cross_interact = json.loads(
    df_binders.loc["HP1", "cross_talk_interaction_energy"].replace("'", "\"")
)["PRC1"]
binders_list[0].chemical_potential = float(cp_HP1)
binders_list[1].chemical_potential = float(cp_PRC1)
binders_list[0].interaction_energy = float(self_interact_HP1)
binders_list[1].interaction_energy = float(self_interact_PRC1)
binders_list[0].cross_talk_interaction_energy["PRC1"] = float(cross_interact)
binders = chromo.binders.make_binder_collection(binders_list)

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
udf_cg = UniformDensityField(
    [p_cg], binders, x_width, nx, y_width, ny, z_width, nz,
    confine_type=confine_type_cg, confine_length=confine_length_cg,
    chi=chi, assume_fully_accessible=assume_fully_accessible,
    fast_field=fast_field
)

# Refine the polymer
n_bind_eq = 1000000
p_refine, udf_refine = rd.refine_chromatin(
    polymer_cg=p_cg,
    num_beads_refined=num_beads,
    bead_spacing=(bead_spacing / (new_cg_factor**3)),
    chemical_mods=chemical_mods,
    udf_cg=udf_cg,
    binding_equilibration=n_bind_eq,
    name_refine="Chr_refine",
    output_dir="output"
)

# Specify move and bead amplitudes
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p_refine])

# Specify simulated annealing schedule
schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "linear_step_for_negative_cp_mild"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

# Run the refined simulation
polymers_refined = mc.polymer_in_field(
    [p_refine],
    binders,
    udf_refine,
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

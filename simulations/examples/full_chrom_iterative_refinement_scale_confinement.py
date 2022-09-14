# Full Chromosome Simulation with One Mark and Dynamic Coarse-Graining

import os
import sys
from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name
from chromo.util.poly_paths import gaussian_walk
import chromo.util.rediscretize as rd
import chromo.util.mu_schedules as ms

# Binders
hp1 = chromo.binders.get_by_name("HP1")
hp1.chemical_potential = -0.4
binders = chromo.binders.make_binder_collection([hp1])

# Confinement
confine_type = "Spherical"
confine_length = 900

# Polymer
num_beads = 393216
bead_spacing = 16.5
chem_mods_path = np.array(["chromo/chemical_mods/meth"])
chemical_mods = Chromatin.load_seqs(chem_mods_path)
states = np.zeros(chemical_mods.shape, dtype=int)
p = Chromatin.confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_length=bead_spacing,
    states=np.zeros(chemical_mods.shape, dtype=int),
    confine_type=confine_type,
    confine_length=confine_length,
    binder_names=np.array(['HP1']),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array(['H3K9me3'])
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

cg_factor = 15

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
    random_seed=np.random.randint(0, 1E5)
)

p_cg = polymers_cg[0]

n_bind_eq = 1000000
p_refine, udf_refine = rd.refine_chromatin(
    polymer_cg=p_cg,
    num_beads_refined=num_beads,
    bead_spacing=bead_spacing,
    chemical_mods=chemical_mods,
    udf_cg=udf_cg,
    binding_equilibration=n_bind_eq,
    name_refine="Chr_refine",
    output_dir="output"
)

amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p_refine])
num_snapshots = 200
mc_steps_per_snapshot = 10000

# Create a list of mu schedules, which are defined in another file
schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "linear_step_for_negative_cp_mild"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

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
    random_seed=np.random.randint(0, 1E5)
)

# Full Chromosome Simulation with Two Marks and Dynamic Coarse-Graining

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
import analyses.adsorption.mu_schedules as ms

# Binders
hp1 = chromo.binders.get_by_name("HP1")
prc1 = chromo.binders.get_by_name("PRC1")
binders = [hp1, prc1]

binders[0].chemical_potential = float(sys.argv[1])
binders[1].chemical_potential = float(sys.argv[2])

self_interaction_HP1 = float(sys.argv[3])
binders[0].interaction_energy = self_interaction_HP1

self_interaction_Polycomb = float(sys.argv[4])
binders[1].interaction_energy = self_interaction_Polycomb

HP1_PRC1_cross_talk_interaction_energy = float(sys.argv[5])
binders[0].cross_talk_interaction_energy["PRC1"] = \
    HP1_PRC1_cross_talk_interaction_energy

binders = chromo.binders.make_binder_collection(binders)

# Confinement
confine_type = "Spherical"
confine_length = 900

# Polymer
num_beads = 393216
bead_spacing = 16.5
chem_mods_path = np.array([
    "chromo/chemical_mods/HNCFF683HCZ_H3K9me3_methyl.txt",
    "chromo/chemical_mods/ENCFF919DOR_H3K27me3_methyl.txt"
])
chemical_mods = Chromatin.load_seqs(chem_mods_path)[:num_beads]
states = np.zeros(chemical_mods.shape, dtype=int)
p = Chromatin.confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_length=bead_spacing,
    states=states,
    confine_type=confine_type,
    confine_length=confine_length,
    binder_names=np.array(['HP1', 'PRC1']),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array(['H3K9me3', 'H3K27me3'])
)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(p.r[:, 0], p.r[:, 1], p.r[:, 2], s=0.5, alpha=0.5)
ax.set_xticks(np.arange(-900, 901, 300))
ax.set_yticks(np.arange(-900, 901, 300))
ax.set_zticks(np.arange(-900, 901, 300))
plt.show()

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

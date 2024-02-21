"""Simulate a chromatin fiber with steric interactions.
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
from inspect import getmembers, isfunction

import chromo.mc as mc
import chromo.polymers as poly
from chromo.polymers import Chromatin
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.util.mu_schedules as ms
from wlcstat.chromo import gen_chromo_conf

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

# Define linker lengths
rev_sim_dir = "/home/users/jwakim/chromo_variable_linkers/simulations/examples"
linker_length_file = os.path.join(
    rev_sim_dir, "example_links_0p09_for_revisions.csv"
)
linker_lengths_bp = np.loadtxt(linker_length_file)
length_bp = 0.332
bp_wrap = 147.
linker_lengths = linker_lengths_bp * length_bp
lp = 150.6024096385542 / length_bp
lt = 301.2048192771084 / length_bp
n_beads = len(linker_lengths) + 1

# Load chemical modifications
chem_mod_file = os.path.join(
    rev_sim_dir, "example_methylation_0p09_for_revisions.csv"
)
chemical_mod_path = np.array([chem_mod_file])
chemical_mods = Chromatin.load_seqs(chemical_mod_path)
modification_name = "H3K9me3"

# Instantiate the HP1 reader protein
binder = chromo.binders.get_by_name('HP1')
binder.chemical_potential = float(sys.argv[1])
binders = chromo.binders.make_binder_collection([binder])

# Binding states
states = np.zeros(chemical_mods.shape, dtype=int)

# Initialize the polymer
p = poly.DetailedChromatinWithSterics.straight_line_in_x(
    "Chr",
    linker_lengths,
    bp_wrap=bp_wrap,
    lp=lp,
    lt=lt,
    binder_names=np.array(["HP1"]),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array([modification_name])
)

# Update positions and orientations using chain growth algorithm
_, _, _, rn, un, orientations = gen_chromo_conf(
    linker_lengths_bp, return_orientations=True
)
t3_temp = orientations["t3_incoming"]
t2_temp = orientations["t2_incoming"]
p.r = rn.copy()
p.r_trial = rn.copy()
p.t3 = t3_temp.copy()
p.t3_trial = t3_temp.copy()
p.t2 = t2_temp.copy()
p.t2_trial = t2_temp.copy()

# Bead density is defined by MacPherson et al. 2018
bead_density = 393216 / (4 / 3 * np.pi * 900 ** 3)
field_length = (n_beads / bead_density / (4 / 3 * np.pi)) ** (1 / 3)

# Specify Field
n_bins_x = int(np.round((63 * field_length) / 900))
n_bins_y = n_bins_x
n_bins_z = n_bins_x
x_width = field_length
y_width = x_width
z_width = x_width

udf = UniformDensityField(
    polymers=[p],
    binders=binders,
    x_width=x_width,
    nx=n_bins_x,
    y_width=y_width,
    ny=n_bins_y,
    z_width=z_width,
    nz=n_bins_z
)
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds([p])
num_snapshots = 200
mc_steps_per_snapshot = 6000

# Create a list of mu schedules, which are defined in another file
schedules = [func[0] for func in getmembers(ms, isfunction)]
select_schedule = "linear_step_for_negative_cp"
mu_schedules = [
    ms.Schedule(getattr(ms, func_name)) for func_name in schedules
]
mu_schedules = [sch for sch in mu_schedules if sch.name == select_schedule]

random_seed = np.random.randint(1, 100000)
output_dir = "output_6"

path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"

p_sim = mc.polymer_in_field(
    [p],
    binders,
    udf,
    mc_steps_per_snapshot,
    num_snapshots,
    amp_bead_bounds,
    amp_move_bounds,
    output_dir=output_dir,
    mu_schedule=mu_schedules[0],
    random_seed=random_seed,
    path_to_run_script=path_to_run_script,
    path_to_chem_mods=chemical_mod_path,
    run_command=run_command
)

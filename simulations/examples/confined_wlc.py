"""Simulate a Confined Wormlike Chain.

Usage:	python confined_wlc.py <CONF_RAD> <N> <SPACING> <LP> <SEED>
Where:	<CONF_RAD> is the radius of the confinement
        <N> is the number of beads in the chain
        <SPACING> is the spacing between two adjacent beads along the chain
        <LP> is the persistence length of the chain
        <SEED> is the random seed for the simulation (OPTIONAL)

By:     Joseph Wakim
Date:   January 29, 2023
Group:  Spakowitz Lab @ Stanford University
"""

# Built-in modules
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)
os.chdir(parent_dir)

print("Root Directory: ")
print(os.getcwd())

# External modules
import numpy as np

# Package modules
import chromo.mc as mc
from chromo.polymers import SSWLC
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name


# Read Commandline arguments
CONF_RAD = float(sys.argv[1])   # Radius of confinement
N = int(sys.argv[2])            # Number of beads in the chain
SPACING = float(sys.argv[3])    # Spacing between neighboring beads
LP = float(sys.argv[4])         # Persistence length of the chain
if len(sys.argv) == 6:
    SEED = int(sys.argv[5])     # Random seed
else:
    SEED = np.random.randint(0, 1E5)
np.random.seed(SEED)

# Store details on simulation
path_to_run_script = os.path.abspath(__file__)
run_command = f"python {' '.join(sys.argv)}"

# Specify reader
null_reader = chromo.binders.get_by_name('null_reader')
binders = chromo.binders.make_binder_collection([null_reader])

# Specify confinement
confine_type = "Spherical"
confine_length = CONF_RAD

# Create Polymer
num_beads = N
bead_spacing = SPACING
lp = LP

polymer = SSWLC.confined_gaussian_walk(
    'poly_1', num_beads, bead_spacing, confine_type=confine_type,
    confine_length=confine_length, binder_names=np.array(['null_reader']), lp=lp
)

# Create Field
n_bins_x = 90
n_bins_y = n_bins_x
n_bins_z = n_bins_x
x_width = 2 * confine_length
y_width = x_width
z_width = x_width
udf = UniformDensityField(
    polymers=[polymer], binders=binders, x_width=x_width, nx=n_bins_x,
    y_width=y_width, ny=n_bins_y, z_width=z_width, nz=n_bins_z,
    confine_type=confine_type, confine_length=confine_length,
    chi=0, vf_limit=1.0, assume_fully_accessible=1
)

# Define simulation
amp_beads, amp_moves = mc.get_amplitude_bounds(polymers = [polymer])
latest_sim = get_unique_subfolder_name("output/sim_")

# Since we are dealing with a homopolymer,
# we do not to simulate reader protein binding/unbinding.
moves_to_use = ctrl.all_moves_except_binding_state(
    log_dir=latest_sim,
    bead_amp_bounds=amp_beads.bounds,
    move_amp_bounds=amp_moves.bounds,
    controller=ctrl.SimpleControl
)
num_snapshots = 250
mc_steps_per_snapshot = 10000
mc.polymer_in_field(
    polymers=[polymer],
    binders=binders,
    field=udf,
    num_save_mc=mc_steps_per_snapshot,
    num_saves=num_snapshots,
    bead_amp_bounds=amp_beads,
    move_amp_bounds=amp_moves,
    output_dir='output',
    mc_move_controllers=moves_to_use,
    random_seed=SEED,
    path_to_run_script=path_to_run_script,
    run_command=run_command
)

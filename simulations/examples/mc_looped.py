"""Run a basic simulation from a script.

This simulation runs a Monte Carlo simulation from a script for easier use of
a remote tower.
"""

# Built-in Modules
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

# External Modules
import numpy as np

# Custom Modules
import chromo
import chromo.mc as mc
from chromo.polymers import LoopedSSTWLC
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name

# Change working directory
os.chdir("../..")
print("Current Working Directory: ")
print(os.getcwd())
print("System path: ")
print(sys.path)

# Specify reader proteins
binders = [chromo.binders.get_by_name('HP1')]
print("Reader Proteins: ")
print(binders)

# Reformat reader proteins into a dataframe format
binders = chromo.binders.make_binder_collection(
    binders
)

# Confine to spherical chrom. territory 1800 um diameter (Cremer & Cremer 2001)
# confine_type = "Spherical"
confine_type = ""
confine_length = 900

print("Constructing polymer...")
# Specify polymers length
num_beads = 1000           # 393217
bead_spacing = 16.5        # About 50 bp linker length

chem_mods_path = np.array(["chromo/chemical_mods/meth"])
chemical_mods = np.zeros((num_beads, 1), dtype=int)

p = LoopedSSTWLC.looped_confined_gaussian_walk(
    'Chr-1',
    num_beads,
    bead_spacing,
    states=chemical_mods.copy(),
    confine_type=confine_type,
    confine_length=confine_length,
    binder_names=np.array(['HP1']),
    chemical_mods=chemical_mods,
    chemical_mod_names=np.array(['H3K9me3'])
)

# Specify the field containing the polymers
n_bins_x = 63
x_width = 2 * confine_length
n_bins_y = n_bins_x
y_width = x_width
n_bins_z = n_bins_x
z_width = x_width
udf = UniformDensityField(
    [p], binders, x_width, n_bins_x, y_width,
    n_bins_y, z_width, n_bins_z, confine_type=confine_type,
    confine_length=confine_length
)


# Specify the bead selection and move amplitude bounds
polymers = [p]
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(polymers)


# Specify MC moves to use
latest_sim = get_unique_subfolder_name("output/sim_")
moves_to_use = ctrl.all_moves_except_binding_state(
    log_dir=latest_sim,
    bead_amp_bounds=amp_bead_bounds.bounds,
    move_amp_bounds=amp_move_bounds.bounds,
    controller=ctrl.SimpleControl
)

if __name__ == "__main__":
    """Run the simulation.
    """
    print("Starting new simulation...")
    num_snapshots = 2
    mc_steps_per_snapshot = 1000
    mc.polymer_in_field(
        [p],
        binders,
        udf,
        mc_steps_per_snapshot,
        num_snapshots,
        amp_bead_bounds,
        amp_move_bounds,
        output_dir='output',
        mc_move_controllers=moves_to_use
    )

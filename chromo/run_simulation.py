# Built-in modules
import os # allows access to operating system functions
import sys # allows access to functions related to the interpreter

# Insert package root to system path
cwd = os.getcwd() # get pathname of current working directory
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir) #gives interpreter a specific path to search

print("Directory containing the notebook:")
print(cwd)

# External modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Package modules
import chromo.mc as mc
from chromo.polymers import SSWLC
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name
import chromo.util.poly_stat as ps
from chromo.test_polymer_build import test_image

# Change working directory to package root
os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

# Instantiate the HP1 reader protein, which is pre-defined in the `chromo.binders` module
null_binder = chromo.binders.get_by_name('null_reader') # must include null binder to make the rest of the code work

# Create the binder collection
binders = chromo.binders.make_binder_collection([null_binder]) #gets relevant binder information from binder specified

num_beads = 1000

#bead_spacing = np.array(variable_linker)
bead_spacing = 25.0 * np.ones((1000, 1)) # change to be real linker lengths later
lp = 100

#Generates the polymer object
polymer = SSWLC.gaussian_walk_polymer(
    'poly_1',
    num_beads,
    bead_spacing,
    lp=lp,
    binder_names=np.array(["null_reader"])
)
sys.exit()
print("test")
# shows a plot of the polymer object
x = polymer.r[:, 0]
y = polymer.r[:, 1]
z = polymer.r[:, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
ax.plot3D(np.asarray(x), np.asarray(y), np.asarray(z))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Tests if the plot shown is sufficiently similar to the reference image.
# control_image_configuration = "control_image_configuration"
# plt.savefig(str("chromo/" + control_image_configuration)) # can be used to generate a new control image

# new_image_name = "testing_plot"
# plt.savefig("chromo/" + str(new_image_name))
# plt.show()

#plt.show()
# test_image(new_image_name, control_image_configuration) # used for comparing image to reference


n_bins_x = 63
n_bins_y = n_bins_x
n_bins_z = n_bins_x

x_width = 1000
y_width = x_width
z_width = x_width

udf = UniformDensityField(
    polymers = [polymer],
    binders = binders,
    x_width = x_width,
    nx = n_bins_x,
    y_width = y_width,
    ny = n_bins_y,
    z_width = z_width,
    nz = n_bins_z
)
amp_bead_bounds, amp_move_bounds = mc.get_amplitude_bounds(
    polymers = [polymer]
)

latest_sim = get_unique_subfolder_name("output/sim_")
moves_to_use = ctrl.all_moves_except_binding_state(
    log_dir=latest_sim,
    bead_amp_bounds=amp_bead_bounds.bounds,
    move_amp_bounds=amp_move_bounds.bounds,
    controller=ctrl.SimpleControl
)

num_snapshots = 200
mc_steps_per_snapshot = 40000

mc.polymer_in_field(
    polymers = [polymer],
    binders = binders,
    field = udf,
    num_save_mc = mc_steps_per_snapshot,
    num_saves = num_snapshots,
    bead_amp_bounds = amp_bead_bounds,
    move_amp_bounds = amp_move_bounds,
    output_dir = 'output',
    mc_move_controllers = moves_to_use
)

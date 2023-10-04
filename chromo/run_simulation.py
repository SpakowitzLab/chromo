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
from chromo.polymers import SSTWLC
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name
import chromo.util.poly_stat as ps
import datetime


pst_time = datetime.datetime.utcnow() + datetime.timedelta(hours=-7)

# Change working directory to package root
os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

# Instantiate the HP1 reader protein, which is pre-defined in the `chromo.binders` module
null_binder = chromo.binders.get_by_name('null_reader') # must include null binder to make the rest of the code work

# Create the binder collection
binders = chromo.binders.make_binder_collection([null_binder]) # gets relevant binder information from binder specified

num_beads = 1000


bead_spacing = np.array([10, 15] * 500)
# print(len(bead_spacing))
# bead_spacing = 15.0 * np.ones((1000, 1)) # change to be real linker lengths later
lp = 100
lt = 100

# Generates the polymer object
"""polymer = SSWLC.gaussian_walk_polymer(
    'poly_1',
    num_beads,
    bead_spacing,
    lp=lp,
    binder_names=np.array(["null_reader"])
)"""

polymer = SSTWLC.gaussian_walk_polymer(
    'poly_1',
    num_beads,
    bead_spacing,
    lp=lp,
    lt=lt,
    binder_names=np.array(["null_reader"]),
)

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

num_snapshots = 560
# num_snapshots = 1000 # try 1000 and average for each set of 100, depending on pre-equilibration steps
# count number of accepted moves for different conditions
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
    mc_move_controllers = moves_to_use,
    temperature_schedule = "no schedule",
    lt_schedule = "logarithmic increase"
)

output_files = os.listdir(latest_sim)

output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("poly_1")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]


all_energies = []
polymer_energies = []

for i, f in enumerate(output_files):
    snap = sorted_snap[i]
    output_path = str(latest_sim) + '/' + f

    r = pd.read_csv(
        output_path,
        header=0,
        skiprows=1,
        usecols=[1, 2, 3],
        dtype=float
    ).to_numpy()

    t3 = pd.read_csv(
        output_path,
        header=0,
        skiprows=1,
        usecols=[4, 5, 6],
        dtype=float
    ).to_numpy()

    polymer.r = r.copy()
    polymer.t3 = t3.copy()

    polymer_energy = polymer.compute_E()
    polymer_energies.append(polymer_energy)

plt.figure(figsize=(8,6))
plt.plot(sorted_snap, polymer_energies)
plt.suptitle("lp = " + str(lp) + ", lt (if used) = " + str(lt) + ", bead spacing = " + str(bead_spacing[1:5]) + " ..." ,
             fontsize = 10)
plt.xlabel("Snapshot number")
plt.ylabel("Polymer Energy")
plt.tight_layout()
plt.savefig(str(latest_sim) + '/' + str(pst_time) + 'Polymer_Energy_Snapshot_Number.png')
#plt.show()


lp = 100    # Persistence length of DNA; in this example, `lp` has no effect
delta = 50  # Monomer monomer separation at which to calculate mean squared distance.
# delta = bead_length/lp

all_dists = []
for i, f in enumerate(output_files):
    snap = sorted_snap[i]
    output_path = str(latest_sim) + '/' + f
    r = pd.read_csv(
        output_path,
        header=0,
        skiprows=1,
        usecols=[1, 2, 3],
        dtype=float
    ).to_numpy()
    poly_stat = ps.PolyStats(r, lp, "overlap")
    windows = poly_stat.load_indices(delta)
    all_dists.append(poly_stat.calc_r2(windows))

plt.figure(figsize=(8, 6))
plt.plot(sorted_snap, all_dists)
plt.xlabel("Snapshot number")
plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
plt.title("Simulation number: " + str(latest_sim))
plt.suptitle("lp = " + str(lp) + ", lt (if used) = " + str(lt) + ", bead spacing = " + str(bead_spacing[1:5]) + " ..." ,
             fontsize = 10)
plt.tight_layout()
plt.savefig(str(latest_sim) + '/' + str(pst_time) + 'R-squared_2lp.png')
#plt.show()


monomer_separation = 10 ** np.arange(-1, 2, 0.05)
monomer_separation = monomer_separation.astype(int)
monomer_separation = np.array(
    [
        monomer_separation[i] for i in range(len(monomer_separation))
        if monomer_separation[i] > 0
    ]
)

monomer_separation_kuhn = monomer_separation * (np.mean(bead_spacing) / lp / 2) # check that this is acceptable


lp = 100 # try 53
kuhn_length = 2 * lp
num_equilibration = 70
all_r2 = []

for i, f in enumerate(output_files):
    if i < num_equilibration:
        continue
    output_path = str(latest_sim) + "/" + f
    r = pd.read_csv(
        output_path,
        header=0,
        skiprows=1,
        usecols=[1, 2, 3],
        dtype=float
    ).to_numpy()
    poly_stat = ps.PolyStats(r, lp, "overlap")
    r2 = []
    for window_size in monomer_separation:
        r2.append(
            poly_stat.calc_r2( # within polystat without kinks
                windows=poly_stat.load_indices(window_size)
            )
        )
    all_r2.append(r2)
all_r2 = np.array(all_r2)
average_squared_e2e = np.mean(all_r2, axis=0)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)


plt.figure(figsize=(8,6), dpi=300)
plt.scatter(np.log10(monomer_separation_kuhn), np.log10(average_squared_e2e), label="simulation")
plt.xlabel(r"Log $L/(2l_p)$")
plt.ylabel(r"Log $\langle R^2 \rangle /(2l_p)^2$")
r2_theory = monomer_separation_kuhn - 1/2 + np.exp(-2 * monomer_separation_kuhn)/2
plt.plot(np.log10(monomer_separation_kuhn), np.log10(r2_theory), label="theory")
plt.legend()
plt.suptitle("lp = " + str(lp) + ", lt (if used) = " + str(lt) + ", bead spacing = " + str(bead_spacing[1:5]) + " ...",
             fontsize = 10)
plt.title("Simulation number: " + str(latest_sim))
plt.suptitle("lp = " + str(lp) + ", lt (if used) = " + str(lt) + ", bead spacing = " + str(bead_spacing[1:5]) + " ..." ,
             fontsize = 10)
plt.tight_layout()
plt.savefig(str(latest_sim) + '/' + str(pst_time) + 'Theory_vs_Simulation.png')
plt.show()

# try 0 as lt
# 25 nm is 75 base pairs
# units of bead spacing are nm

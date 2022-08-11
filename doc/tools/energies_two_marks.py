"""Calcualate System Energies.

Joseph Wakim
August 20, 2021
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chromo.polymers as poly
import chromo.fields as fld
import chromo.binders
import chromo.util.poly_stat as ps

# Navigate to output directory
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

# Specify simulation to evaluate
# sim = ps.get_latest_simulation()
sim_id = 18
sim = f"sim_{sim_id}"
delta = 50
lp = 53
print("Sim: " + sim)

# List output files in sorted order
output_dir = os.getcwd() + '/' + sim
output_files = os.listdir(output_dir)
output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
#    f for f in output_files if f.endswith(".csv") and f.startswith("poly")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]

output_path = sim + "/" + output_files[0]

r = np.ascontiguousarray(pd.read_csv(
    output_path, usecols=[1, 2, 3], skiprows=[0, 1]
).to_numpy())
n_beads = len(r)
t3 = np.ascontiguousarray(pd.read_csv(
    output_path, usecols=[4, 5, 6], skiprows=[0, 1]
).to_numpy())
states = np.ascontiguousarray(pd.read_csv(
    output_path, usecols=[10,11], skiprows=[0, 1]
).to_numpy())
chemical_mods = np.ascontiguousarray(np.zeros((len(r), 1), dtype=int))
binder_name = np.array(["HP1", "PRC1"])
chemical_mod_name = np.array(["H3K9me3", "H3K27me3"])

# Instantiate Polymer Object
confine_length = 364.93211973440407
frac_full_chromo = n_beads / 393216
confine_type = "Spherical"
# confine_length *= np.cbrt(frac_full_chromo)
polymer = poly.Chromatin(
    "polymer",
    r,
    bead_length=53,
    t3=t3,
    t2=t3,
    states=states,
    binder_names=binder_name,
    chemical_mods=chemical_mods,
    chemical_mod_names=chemical_mod_name,
)

# Instantiate Field Object
# n_bins_x = int(round(63 * np.cbrt(frac_full_chromo)))
n_bins_x = 27
# x_width = 2 * confine_length
x_width = 771.4285714285713
n_bins_y = n_bins_x
y_width = x_width
n_bins_z = n_bins_x
z_width = x_width

binder_objs = [chromo.binders.get_by_name('HP1'), chromo.binders.get_by_name('PRC1')]
binder_objs[0].chemical_potential = -0.4
binder_objs[1].chemical_potential = -0.4
binder_objs[0].cross_talk_interaction_energy["PRC1"] = -0.5

binders = chromo.binders.make_binder_collection(binder_objs)
field = fld.UniformDensityField(
    [polymer], binders, x_width, n_bins_x, y_width, n_bins_y, z_width, n_bins_z,
    confine_type = confine_type, confine_length = confine_length
)

# Calculate Polymer Energies
all_energies = []
polymer_energies = []
field_energies = []

for i, f in enumerate(output_files):
    print(f)
    snap = sorted_snap[i]
    output_path = sim + '/' + f

    r = pd.read_csv(
        output_path,
        skiprows=2,
        usecols=[1, 2, 3],
        dtype=float
    ).to_numpy()

    t3 = pd.read_csv(
        output_path,
        skiprows=2,
        usecols=[4, 5, 6],
        dtype=float
    ).to_numpy()

    states = pd.read_csv(
        output_path,
        skiprows=2,
        usecols=[10, 11],
        dtype=int
    ).to_numpy()

    polymer.r = r.copy()
    polymer.t3 = t3.copy()
    polymer.states = states.copy()

    field_energy = field.compute_E(polymer)
    polymer_energy = polymer.compute_E()
    polymer_energies.append(polymer_energy)
    field_energies.append(field_energy)
    all_energies.append(polymer_energy + field_energy)

# Remove high energy indices
"""
high_energy_inds = set([
    ind for ind in range(len(all_energies))
    if all_energies[ind] >= 1E90
])
sorted_snap = np.array([
    sorted_snap[ind] for ind in range(len(sorted_snap))
    if ind not in high_energy_inds
])
field_energies = np.array([
    field_energies[ind] for ind in range(len(field_energies))
    if ind not in high_energy_inds
])
polymer_energies = np.array([
    polymer_energies[ind] for ind in range(len(polymer_energies))
    if ind not in high_energy_inds
])
all_energies = np.array([
    all_energies[ind] for ind in range(len(all_energies))
    if ind not in high_energy_inds
])
"""

plt.figure()
plt.plot(sorted_snap, all_energies, color='k', label='Total')
plt.plot(sorted_snap, field_energies, color='r', label='Field')
plt.plot(sorted_snap, polymer_energies, color='b', label='Polymer')
plt.xlabel("Snapshot number")
plt.ylabel(r"Energy ($kT$)")
plt.legend()
plt.tight_layout()
plt.savefig(sim + "/Energy_Evolution.png", dpi=600)
plt.close()

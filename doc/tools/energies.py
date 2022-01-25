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
import chromo.marks
import chromo.util.poly_stat as ps

# Navigate to output directory
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

# Specify simulation to evaluate
# sim = ps.get_latest_simulation()
sim = "sim_57"
delta = 50
lp = 53
print("Sim: " + sim)

# List output files in sorted order
output_dir = os.getcwd() + '/' + sim
output_files = os.listdir(output_dir)
output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
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
    output_path, usecols=[10], skiprows=[0, 1]
).to_numpy())
chemical_mods = np.ascontiguousarray(np.zeros((len(r), 1), dtype=int))
mark_name = np.array(["HP1"])
chemical_mod_name = np.array(["H3K9me3"])

# Instantiate Polymer Object
polymer = poly.Chromatin(
    "polymer",
    r,
    bead_length=53,
    t3=t3,
    t2=t3,
    states=states,
    mark_names=mark_name,
    chemical_mods=chemical_mods,
    chemical_mod_names=chemical_mod_name
)

# Instantiate Field Object
confine_length = 900
n_bins_x = 63
x_width = 2 * confine_length
n_bins_y = n_bins_x
y_width = x_width
n_bins_z = n_bins_x
z_width = x_width
mark_objs = [chromo.marks.get_by_name('HP1')]
marks = chromo.marks.make_mark_collection(mark_objs)
field = fld.UniformDensityField(
    [polymer], marks, x_width, n_bins_x, y_width, n_bins_y, z_width, n_bins_z
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
        usecols=[10],
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

plt.figure()
plt.plot(sorted_snap, all_energies, color='k', label='Total')
plt.plot(sorted_snap, field_energies, color='r', label='Field')
plt.plot(sorted_snap, polymer_energies, color='b', label='Polymer')
plt.xlabel("Snapshot number")
plt.ylabel("Polymer Energy")
plt.legend()
plt.tight_layout()
plt.savefig(sim + "/Energy_Evolution.png", dpi=600)
plt.close()

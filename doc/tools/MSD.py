"""Generate RMSD for finite bead separation.

For a specified bead spacing, calculate the root-mean-squared end-to-end
distance for beads specified by that spacing in each snapshot.

Joseph Wakim
August 19, 2021
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chromo.util.poly_stat as ps

# Navigate to output directory
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

# Specify simulation to evaluate
# sim = ps.get_latest_simulation()
sim = "sim_338"
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

# Collect end-to-end distances
all_dists = []
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
    poly_stat = ps.PolyStats(r, lp, "overlap")
    windows = poly_stat.load_indices(delta)
    all_dists.append(poly_stat.calc_r2(windows))

plt.figure()
plt.plot(sorted_snap, all_dists)
plt.xlabel("Snapshot number")
plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
plt.tight_layout()
plt.savefig(sim + "/MSD_Evolution.png", dpi=600)
plt.close()

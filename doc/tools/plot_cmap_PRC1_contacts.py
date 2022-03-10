"""Generate contact map from configuration file.

Generate heat map representing the probability of contact between pairs of
nucleosome positions.
"""

import os
import sys
import csv

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chromo.util.poly_stat as ps
from chromo.fields import (
    assign_beads_to_bins, get_neighboring_bins, get_blocks
)

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

#sim = ps.get_latest_simulation()
sim = "sim_" + str(int(sys.argv[1]))
#num_equilibration_steps = 50

print("Sim: " + sim)
output_dir = os.getcwd() + '/' + sim
output_files = os.listdir(output_dir)
output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
max_snapshot = np.max(snapshot)
num_equilibration_steps = max_snapshot-20

sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]
output_files = [
    output_files[i] for i in range(len(output_files))
    if sorted_snap[i] > num_equilibration_steps - 1 and
    sorted_snap[i] < num_equilibration_steps + 10
]
os.chdir(parent_dir)

x_width = 1800
nx = 63
y_width = 1800
ny = 63
z_width = 1800
nz = 63
block_size = 50
cutoff_dist = 10000

print("Loading Polymers")
for i, f in enumerate(output_files):
    if (i+1) % 10 == 0:
        print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
        print()
    output_path = "output/" + sim + "/" + f
    r = np.ascontiguousarray(
        pd.read_csv(output_path, usecols=[1, 2, 3, 11], skiprows=[0, 1]).to_numpy()
    )
    if i == 0:
        n_beads = len(r)
        blocks = get_blocks(n_beads, block_size)
        num_blocks = int(np.ceil(n_beads / block_size))
        contacts = np.zeros((num_blocks, num_blocks))
        contacts_at_sep = np.zeros(n_beads)

    bins = assign_beads_to_bins(
        np.ascontiguousarray(r[:, :3]), n_beads, nx, ny, nz, x_width, y_width, z_width
    )
    n_bins = len(bins.keys())
    neighboring_bins = get_neighboring_bins(nx, ny, nz)

    print("Collecting contacts from file: " + str(f))
    for i in bins.keys():
        if (i+1) % 1000 == 0:
            print("Bin " + str(i+1) + " of " + str(n_bins))

        for nbr_bin in neighboring_bins[i]:
            beads_in_bin = len(bins[nbr_bin])

            for k in range(beads_in_bin):
                for l in range(k, beads_in_bin):
                    ind_1 = bins[nbr_bin][k]
                    ind_2 = bins[nbr_bin][l]

                    block_1 = blocks[ind_1]
                    block_2 = blocks[ind_2]
                    wt = min(r[ind_1, 3], r[ind_2, 3])

                    diff = r[ind_1, :3] - r[ind_2, :3]
                    dist = np.linalg.norm(diff)

                    if dist <= cutoff_dist:
                        contacts[block_1, block_2] += 1 * wt
                        contacts[block_2, block_1] += 1 * wt
                        sep = np.abs(ind_2 - ind_1)
                        contacts_at_sep[sep] += 1

contacts /= len(output_files)
log_contacts = np.log10(contacts+1)
contacts_at_sep /= len(output_files)

print("Saving contact matrix...")
np.savetxt(output_dir + "/contact_matrix_single_sim_PRC1.csv", contacts, delimiter=",")
np.savetxt(output_dir + "/contacts_at_sep_single_sim_PRC1.csv", contacts_at_sep, delimiter=",")

font = {'family' : 'serif',
        'weight':'normal',
        'size': 18}
plt.rc('font', **font)

print("Plotting contact map...")
plt.figure()
plt.imshow(log_contacts, cmap='Reds')
plt.colorbar()
plt.xlabel("Locus One (kb)")
plt.ylabel("Locus Two (kb)")
plt.tight_layout()
plt.savefig(output_dir + "/contact_matrix_single_sim_PRC1.png", dpi=600)

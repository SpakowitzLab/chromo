"""Generate contact maps from simulation outputs.

Author: Joseph Wakim
Group:  Spakowitz Lab
Date:   August 24, 2022
Usage:  python plot_cmap_new.py <SIM_ID>
"""

# Built-in Modules
import os
import sys
import csv
from pprint import pprint

# Third Party Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add root directory to path
cwd = os.getcwd()
parent_dir = f"{cwd}/../.."
sys.path.insert(1, parent_dir)

# Custom modules
import chromo.util.rediscretize as rd
import chromo.fields as fd

# Load simulation and navigate to simulation output directory
sim_id = sys.argv[1]
os.chdir(f"{parent_dir}/output/sim_{sim_id}")

# Specify contact map to plot
output_prefix = "Chr_CG"
n_equilibrate = 180
field_path = "UniformDensityField"
cg_factor = 5     # 100 or 50 works well for refined; 5 works well for CG

# To plot the contact map of a specific snapshot, specify below
specific_snap = False
specific_snap_ind = 0

# Load field parameters
field_params = pd.read_csv(
    field_path, header=None, names=["Attribute", "Value"], index_col=0
)
x_width = float(field_params.loc["x_width", "Value"])
y_width = float(field_params.loc["y_width", "Value"])
z_width = float(field_params.loc["z_width", "Value"])
nx = int(field_params.loc["nx", "Value"])
ny = int(field_params.loc["ny", "Value"])
nz = int(field_params.loc["nz", "Value"])
nx = round(nx / 3)     # 7.5 works well for refined; 3 works well for CG
ny = round(ny / 3)     # 7.5 works well for refined; 3 works well for CG
nz = round(nz / 3)     # 7.5 works well for refined; 3 works well for CG
n_bins = nx * ny * nz

# Load snapshots
files = os.listdir()
snapshots = [
    file for file in files
    if file.startswith(output_prefix) and file.endswith(".csv")
]
snap_inds = [int(snap.split("-")[-1].split(".")[0]) for snap in snapshots]
snapshots = [snap for _, snap in sorted(zip(snap_inds, snapshots))]
snap_inds = np.sort(snap_inds)

# Filter snapshots based on equilibration or specific index
if not specific_snap:
    snapshots_filtered = [
        snapshots[i] for i in range(len(snapshots))
        if snap_inds[i] > n_equilibrate
    ]
else:
    snapshots_filtered = [
        snapshots[i] for i in range(len(snapshots))
        if snap_inds[i] == specific_snap_ind
    ]

# Determine contact probabilities
n_snapshots = len(snapshots_filtered)
weight = 1 / n_snapshots * cg_factor
for i, snap in enumerate(snapshots_filtered):
    print(f"Snap: {snap}")
    r = pd.read_csv(
        snap, usecols=[1, 2, 3], skiprows=[0, 1], header=None
    ).to_numpy()
    if i == 0:
        n_beads_full_res = len(r)
        group_intervals = rd.get_cg_bead_intervals(n_beads_full_res, cg_factor)
        n_beads_cg = len(group_intervals)
        contact_map = np.zeros((n_beads_cg, n_beads_cg), dtype=float)
        nbr_bins = fd.get_neighboring_bins(nx, ny, nz)
    r_cg = rd.get_avg_in_intervals(r, group_intervals)
    bin_map = fd.assign_beads_to_bins(
        np.ascontiguousarray(r_cg), n_beads_cg, nx, ny, nz, x_width, y_width,
        z_width
    )
    for j in range(n_bins):
        for k in nbr_bins[j]: 
            # Do not double-count neighbors 
            if k < j:
                continue
            for ind0 in bin_map[j]:
                for ind1 in bin_map[k]:
                    contact_map[ind0, ind1] += weight
                    contact_map[ind1, ind0] += weight
log_contacts = np.log10(contact_map+1)

# Save contact matrix
np.savetxt("log_contact_matrix.csv", log_contacts, delimiter=",")

# Plot contact map
font = {'family' : 'serif',
        'weight':'normal',
        'size': 24}
plt.rc('font', **font)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.5)
extents = [0, 90, 90, 0]
im = ax.imshow(log_contacts, cmap="Reds", extent=extents)
ax.set_xticks([15, 30, 45, 60, 75])
ax.set_yticks([15, 30, 45, 60, 75])
ticks = np.arange(0, np.ceil(np.max(log_contacts)), 1)
boundaries = np.linspace(0, np.ceil(np.max(log_contacts)), 1000)
ax.set_xlabel("Locus One (Mb)")
ax.set_ylabel("Locus Two (Mb)")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
fig.colorbar(
    im, cax=cax, orientation='vertical', ticks=ticks, boundaries=boundaries
)
plt.tight_layout()
plt.savefig("log_contact_matrix.png", dpi=600)

# Plot contact map as represented by MacPherson et al.
font = {'family' : 'serif',
        'weight':'normal',
        'size': 24}
plt.rc('font', **font)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.5)
extents = [5, 35, 35, 5]
max_ind = len(log_contacts)-1
lower_bound = int(round((5 / 90) * max_ind))
upper_bound = int(round((35 / 90) * max_ind))
im = ax.imshow(
    log_contacts[lower_bound:upper_bound, lower_bound:upper_bound],
    cmap="Reds", extent=extents
)
ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
ax.set_yticks([5, 10, 15, 20, 25, 30, 35])
ticks = np.arange(0, np.ceil(np.max(log_contacts)), 1)
boundaries = np.linspace(0, np.ceil(np.max(log_contacts)), 1000)
ax.set_xlabel("Chr 16 (Mb)")
ax.set_ylabel("Chr 16 (Mb)")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
fig.colorbar(
    im, cax=cax, orientation='vertical', ticks=ticks, boundaries=boundaries
)
plt.tight_layout()
plt.savefig("log_contact_matrix_MacPherson.png", dpi=600)

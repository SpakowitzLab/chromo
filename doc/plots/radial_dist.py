"""Generate distribution of radial positions.

Joseph Wakim
October 13, 2021
"""

import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

# sim = ps.get_latest_simulation()
sim = "sim_3"

print("Sim: " + sim)
output_dir = os.getcwd() + '/' + sim
output_files = os.listdir(output_dir)
output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]
last_output_files = output_files[-100:]

os.chdir(parent_dir)

counts_all = []
for last_output_file in last_output_files:
    output_path = "output/" + sim + "/" + last_output_file

    r = pd.read_csv(
            output_path, usecols=[1, 2, 3], skiprows=[0]
        ).dropna().to_numpy()

    radial_dists = np.linalg.norm(r, axis=1)

    step_size = 100
    bins = np.arange(100, 4501, step_size)
    counts, bin_edges = np.histogram(radial_dists, bins=bins)
    counts = counts.astype(float)
    counts_all.append(counts)

counts_all = np.array(counts_all)
counts_avg = np.sum(counts_all, axis=0)

# Correct densities based on volumes of spherical shells
for i in range(len(bin_edges)-1):
    volume = 4/3 * np.pi * ((bin_edges[i+1]/1E3)**3 - (bin_edges[i]/1E3)**3)
    counts_avg[i] /= volume

counts_avg /= np.sum(counts_avg)

# Get theoretical radial densities
a = 4500
b = 53
N = len(r)
r_theory = np.arange(100, 4501, 1)
n_max = 1000
rho = np.zeros(len(r_theory))
for n in range(2, n_max + 1):
    rho += (-1)**(n+1) / (n * np.pi) * np.sin(np.pi * r_theory / a) * np.sin(n * np.pi * r_theory / a) / (r_theory**2 * b**2 * (n**2 - 1))
rho += N / (np.pi) * np.sin(np.pi * r_theory / a)**2 / r_theory**2

normalize = np.sum(rho)
rho_theory = rho / normalize * step_size


print(rho_theory)

print("Counts:")
print(counts_avg)
print("Bin Edges:")
print(bin_edges)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18}

plt.rc('font', **font)
os.chdir(output_dir)
plt.figure(figsize=(8, 6))
plt.hist(bin_edges[:-1], bin_edges, weights=counts_avg)
plt.plot(r_theory, rho_theory)
plt.xlabel("Radial Distance (nm)")
plt.ylabel(r"Probability")
plt.tight_layout()
plt.savefig("radial_distances.png", dpi=600)

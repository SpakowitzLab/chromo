
import os
import sys

cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chromo.util.poly_stat as ps
import chromo.polymers as polymers

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd + '/../../output')

sim = ps.get_latest_simulation()
# sim = "sim_32"
num_equilibration_steps = 3800

print("Sim: " + sim)
output_dir = os.getcwd() + '/' + sim
output_files = os.listdir(output_dir)
output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("Chr")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]
output_files = [
    output_files[i] for i in range(len(output_files))
    if sorted_snap[i] > num_equilibration_steps - 1
]

os.chdir(parent_dir)

log_vals = np.arange(-1, 2, 0.05)
bead_range = 10 ** log_vals   # * 80
bead_range = bead_range.astype(int)
bead_range = np.array(
    [bead_range[i] for i in range(len(bead_range)) if bead_range[i] > 0]
)
bead_range = np.unique(bead_range)

print("Loading polymers")
all_r2 = []
for i, f in enumerate(output_files):
    if (i+1) % 10 == 0:
        print("Snapshot: " + str(i+1) + " of " + str(len(output_files)))
        print()
    output_path = "output/" + sim + "/" + f
    r = pd.read_csv(
    	output_path, usecols=[1, 2, 3], skiprows=[0, 1]
    ).dropna().to_numpy()
    lp = 53
    # Retrieve the kuhn length once, since it remains the same across snapshots
    if i == 0:
        kuhn_length = 2 * lp
    poly_stat = ps.PolyStats(r, lp, "overlap")
    r2 = []
    for j, window_size in enumerate(bead_range):
        r2.append(
            poly_stat.calc_r2(
                windows=poly_stat.load_indices(window_size)
            )
        )
    all_r2.append(r2)
all_r2 = np.array(all_r2)
average_squared_e2e = np.mean(all_r2, axis=0)

bead_range = bead_range / 2     # Convert x axis to number of kuhn lengths

os.chdir(output_dir)
with open("avg_squared_e2e.txt", "w") as output_file:
    for val in average_squared_e2e:
        output_file.write('%s\n' % val)

plt.figure()
plt.scatter(bead_range, average_squared_e2e)
plt.xlabel(r"$L/(2l_p)$")
plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
plt.yscale("log")
plt.xscale("log")
plt.savefig("Squared_e2e_vs_dist_v2.png", dpi=600)
os.chdir(cwd)

# Plot the mean squared end-to-end distance on a log-log plot

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

os.chdir(output_dir)
plt.figure()
plt.scatter(np.log10(bead_range), np.log10(average_squared_e2e))
plt.xlabel(r"Log $L/(2l_p)$")
plt.ylabel(r"Log $\langle R^2 \rangle /(2l_p)^2$")
r2_theory = 2 * (bead_range / 2 - (1 - np.exp(-(2) * bead_range)) / (2) ** 2)
plt.plot(np.log10(bead_range), np.log10(r2_theory))
plt.tight_layout()
plt.savefig("Log_Log_Squared_e2e_vs_dist_v2.png", dpi=600)
os.chdir(cwd)

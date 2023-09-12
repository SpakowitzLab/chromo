# Built-in modules
import os # allows access to operating system functions
import sys # allows access to functions related to the interpreter

# Insert package root to system path
cwd = os.getcwd() # get pathname of current working directory
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir) # gives interpreter a specific path to search

print("Directory containing the notebook:")
print(cwd)

# External modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chromo.util.poly_stat as ps
import chromo.util.mc_stat as mc_stat


# Change working directory to package root
os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())

"""

num_beads = 1000


bead_spacing = np.array([15, 25] * 500)
lp = 100    # Persistence length of DNA; in this example, `lp` has no effect
delta = 20  # Monomer monomer separation at which to calculate mean squared distance.
lt = 100
"""
# Load names of polymer configuration output files
# 206 is with 200 snapshots of twist with alternating 15 and 25
# 130 is without twist with alternating 15 and 25
latest_sim = "output/sim_206"

output_files = os.listdir(latest_sim)

output_files = [
    f for f in output_files if f.endswith(".csv") and f.startswith("poly_1")
]
snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
sorted_snap = np.sort(np.array(snapshot))
output_files = [f for _, f in sorted(zip(snapshot, output_files))]

break_boundaries = [0, 1000]
"""
all_dists = []
for i, f in enumerate(output_files):
    if i < break_boundaries[0]:
        continue
    if i > break_boundaries[1]:
        break
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
    print("this is what window looks like")
    print(windows[0])

    all_dists.append(poly_stat.calc_r2(windows))
#plt.figure(figsize=(8, 6))
#plt.plot(sorted_snap, all_dists)
#plt.xlabel("Snapshot number")
#plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
#plt.title("Simulation number: " + latest_sim)
#plt.suptitle("lp = " + str(lp) + ", lt (if used) = " + str(lt) + ", bead spacing = " + str(bead_spacing[1:5]) + " ..." ,
 #            fontsize = 10)
plt.tight_layout()
plt.show()

monomer_separation = 10 ** np.arange(-1, 2, 0.05)
monomer_separation = monomer_separation.astype(int)
monomer_separation = np.array(
    [
        monomer_separation[i] for i in range(len(monomer_separation))
        if monomer_separation[i] > 0
    ]
)

monomer_separation_kuhn = monomer_separation * (np.mean(bead_spacing) / lp / 2)



kuhn_length = 2 * lp
num_equilibration = 70
all_r2 = []



for i, f in enumerate(output_files):
    if i < break_boundaries[0]:
        continue
    if i > break_boundaries[1]:
        break
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
plt.suptitle("lp = " + str(lp) + ", lt (if used) = " + str(lt) + ", bead spacing = " +", lower sim number bound " + str(break_boundaries[0]) +", upper sim number bound: " + str(break_boundaries[1])+ " bead spacing "+ str(bead_spacing[1:5]) + " ...",
             fontsize = 10)
plt.title("Simulation number: " + latest_sim)
plt.tight_layout()
plt.show()

ratio = average_squared_e2e/r2_theory
"""
#print(ratio)

def find_RMSD(array1, array2):

    RMSD = np.sqrt(
        1 / array1.shape[0] * np.sum(
            np.linalg.norm(array2 - array1, ord=1, axis=1)
            )
        )

    return RMSD

num_equilibration = 70
RMSD_list = []
x_list = []
original_configuration = []
x = 0
for i, f in enumerate(output_files):
    if i < break_boundaries[0]:
        continue
    if i > break_boundaries[1]:
        break
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
    if len(RMSD_list) == 0:
        RMSD_list.append(find_RMSD(r, r))
        original_configuration = r
    else:
        RMSD_list.append(find_RMSD(original_configuration, r))
    #original_configuration = r
    x_list.append(x)
    x = x + 1

plt.scatter(x_list, RMSD_list, label="RMSD progression")
plt.show()
"""
xlog_data = np.log(x_list)

curve = np.polyfit(xlog_data, RMSD_list, 1)
print(curve)
"""
    #mc_stat.ConfigurationTracker.log_move()




from scipy.optimize import curve_fit

# Define the logarithmic function
def logarithmic_curve(x, a, b):
    return a * np.log(x) + b # natural log

# Fit the curve to the data
x_list = x_list[1:]
RMSD_list = RMSD_list[1:]
popt, _ = curve_fit(logarithmic_curve, x_list, RMSD_list)

# The optimal parameters (coefficients)
a_optimal, b_optimal = popt

# Print the coefficients
print("Coefficient a (a_optimal):", a_optimal)
print("Coefficient b (b_optimal):", b_optimal)

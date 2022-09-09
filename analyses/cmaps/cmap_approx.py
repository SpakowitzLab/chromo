"""Generate approximate contact maps from simulation outputs.

Usage:      python cmap_approx.py <OUTPUT_DIR> <SIM_IND> <NUM_EQUILIBRATION>

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       August 24, 2022
"""

import sys

import analyses.cmaps.cmaps as cm
import analyses.characterizations.inspect_simulations as inspect


if __name__ == "__main__":
    
    # Parse command line arguments
    output_dir = sys.argv[1]
    sim_ind = int(sys.argv[2])
    num_equilibration = int(sys.argv[3])
    
    # Determine whether simulation is coarse-grained or refined
    # Assume the simulation is refined unless otherwise specified
    snap_paths = get_snapshot_paths(
        output_dir, sim_ind, num_equilibration
    )
    if "cg" in snap_paths[0].split("/")[-1].split(".")[0].lower():
        print(f"Analyzing `sim_{sim_ind}` as coarse grained simulation.")
        cg_factor = 5
        resize_factor = 3
    else:
        print(f"Analyzing `sim_{sim_ind}` as a refined simulation.")
        cg_factor = 100
        resize_factor = 7.5
    
    # Generate the contact map
    cm.plot_approx_cmap(
        output_dir, sim_ind, cg_factor, resize_factor, num_equilibration
    )

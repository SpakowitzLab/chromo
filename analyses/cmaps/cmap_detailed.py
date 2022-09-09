"""Generate a detailed contact map from simulation outputs

Usage:      python cmap_detailed.py <OUTPUT_DIR> <INTEGER_SIM_IND> <NUM_EQUILIBRATION> <FLOAT_NBR_CUTOFF>

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford University
Date:       September 3, 2022
"""

import sys

import analyses.cmaps.cmaps as cm


if __name__ == "__main__":
    
    # Parse command line arguments
    output_dir = sys.argv[1]
    sim_ind = int(sys.argv[2])
    num_equilibration = int(sys.argv[3])
    nbr_cutoff = float(sys.argv[4])
    
    # Generate the contact map
    cm.plot_detailed_cmap(
        output_dir, sim_ind, num_equilibration, nbr_cutoff
    )

"""Calculate the MSDs of a polymer over the course of a simulation.

Usage:      python MSDs.py <OUTPUT_DIR> <INT_SIM_INDEX> <SEG_LENGTH> <PERSISTENCE_LENGTH> <SAVE_FILE>

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 7, 2022

"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chromo.util.poly_stat as ps
import analyses.characterizations.inspect_simulations as inspect


def get_MSDs(
    output_dir: str, sim_ind: int, seg_length: int, lp: float,
    polymer_prefix: Optional[str] = "Chr", sim_prefix: Optional[str] = "sim_",
) -> np.ndarray:
    """Get the MSD for a specified segment length at each simulation snapshot.

    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    seg_length : int
        Number of beads in the segment for which MSD is to be computed
    lp : float
        Persistence length of polymer (for normalization)
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
        (default = "Chr")
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")

    Returns
    -------
    np.ndarray (N,) of float
        Array of MSDs for specified segment lengthat each configurational
        snapshot.
    """
    snap_paths = inspect.get_snapshot_paths(
        output_dir, sim_ind, polymer_prefix=polymer_prefix,
        sim_prefix=sim_prefix, num_equilibration=0
    )
    all_MSDs = []
    for i, path in enumerate(snap_paths):
        r = pd.read_csv(
            path, skiprows=2, usecols=[1, 2, 3], dtype=float, header=None
        ).to_numpy()
        poly_stat = ps.PolyStats(r, lp, "overlap")
        windows = poly_stat.load_indices(seg_length)
        all_MSDs.append(poly_stat.calc_r2(windows))
    return np.array(all_MSDs)


def plot_MSDs(
    output_dir: str, sim_ind: int, seg_length: int, lp: float, save_file: str,
    polymer_prefix: Optional[str] = "Chr", sim_prefix: Optional[str] = "sim_"
):
    """Plot the MSD for a specified segment length at each simulation snapshot.

    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    seg_length : int
        Number of beads in the segment for which MSD is to be computed
    lp : float
        Persistence length of polymer (for normalization)
    save_file : str
        File name at which to save profile of MSDs; saved to the
        simulation output directory
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
        (default = "Chr")
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
    """
    all_MSDs = get_MSDs(
        output_dir, sim_ind, lp, seg_length, polymer_prefix, sim_prefix
    )
    sorted_snaps = np.arange(len(all_MSDs))
    plt.figure()
    plt.plot(sorted_snaps, all_MSDs)
    plt.xlabel("Snapshot number")
    plt.ylabel(r"$\langle R^2 \rangle /(2l_p)^2$")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sim_prefix}{sim_ind}/{save_file}", dpi=600)
    plt.close()


def main(output_dir, sim_ind, seg_lenght, lp, save_file):
    """Plot MSDs over the course of a simulation.
    
    Notes
    -----
    See documentation for `plot_MSDs`.
    """
    plot_MSDs(output_dir, sim_ind, seg_length, lp, save_file)


if __name__ == "__main__":
    
    output_dir = sys.argv[1]
    sim_ind = int(sys.argv[2])
    seg_length = int(sys.argv[3])
    lp = float(sys.argv[4])
    save_file = sys.argv[5]
    
    print("Evaluating and plotting MSDs...")
    main(output_dir, sim_ind, seg_length, lp, save_file)

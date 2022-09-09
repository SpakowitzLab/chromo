"""Evaluate the mean-squared end-to-end distance of internal polymer segments.

Notes
-----
This module provides utility functions, but is not intended to be run
independently

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 6, 2022
"""

# Internal Modules
import os
from typing import List, Optional

# Third Party Modules
import numpy as np
import pandas as pd

# Custom Modules
import chromo.util.poly_stat as ps
from analyses.characterizations.inspect_simulations import get_snapshot_paths


def get_interval_widths(
    lower_log: float, upper_log: float, log_step: float
) -> np.ndarray:
    """Get bead interval widths that are linearly separated on a log scale.

    Notes
    -----
    Generate linearly spaced values on a log scale. Then, invert the log to
    get the same values on a linear scale. Approximate the values on the
    linear scale as integers, then remove duplicates and negatives. This
    function gives roughly evenly spaced values on a log-transformed axis.

    Parameters
    ----------
    lower_log, upper_log : float
        Lower and upper bounds of the bead interval widths on a log scale
    log_step : float
        Spacing of the bead interval widths on a log scale

    Returns
    -------
    np.ndarray (N,) of int
        Bead interval widths that are approximately evenly spaced on a log
        scale
    """
    log_vals = np.arange(lower_log, upper_log, log_step)
    bead_range = 10 ** log_vals
    bead_range = bead_range.astype(int)
    bead_range = np.array(
        [bead_range[i] for i in range(len(bead_range)) if bead_range[i] > 0]
    )
    bead_range = np.unique(bead_range)
    return bead_range


def get_mean_r2(
    output_paths: List[str], bead_range: np.ndarray, lp: float
) -> np.ndarray:
    """Get mean squared end-to-end distance for variable width bead intervals.

    Parameters
    ----------
    output_paths : List[str]
        List of paths to equilibrated configurational snapshots
    bead_range : np.ndarray (N,) of int
        Bead interval widths for evaluation of mean r2
    lp : float
        Persistence length of the polymer for nondimensionalization

    Returns
    -------
    np.ndarray (N,) of float
        Mean squared end-to-end distance of polymer segments with widths defined
        in bead_range, nondimensionalized by squared kuhn length
    """
    all_r2 = []
    for i, output_path in enumerate(output_paths):
        if (i+1) % 10 == 0:
            print(f"Snapshot: {i+1} of {len(output_paths)}\n")
        r = pd.read_csv(
            output_path, usecols=[1, 2, 3], header=None, skiprows=[0, 1]
        ).dropna().to_numpy()
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
    avg_r2 = np.mean(all_r2, axis=0)
    return avg_r2


def get_interval_widths_kuhn(
    bead_range: np.ndarray, bead_spacing: float, lp: float
) -> np.ndarray:
    """Re-express bead intervals from index sep. to nondimens. segment length.

    Parameters
    ----------
    bead_range : np.ndarray (N,) of int
        Bead interval widths for evaluation of mean r2
    bead_spacing : float
        Segment length separating adjacent beads in the polymer
    lp : float
        Persistence length of the polymer for nondimensionalization

    Returns
    -------
    np.ndarray (N,) of float
        Bead interval widths expressed as segment lengths nondimensionalized
        by the Kuhn length of the polymer
    """
    return bead_range * bead_spacing / (2 * lp)


def get_avg_r2_theory(seg_lengths_kuhn: np.ndarray) -> np.ndarray:
    """Get theoretical, nondimensionalized mean squared end-to-end distances.

    Notes
    -----
    This function returns the theoretical mean squared end-to-end distances
    of variable length internal segments of the polymer, according to the
    wormlike chain model. The mean squared end-to-end distances are
    nondimensionalized by kuhn length squared.

    Parameters
    ----------
    seg_lengths_kuhn : np.ndarray (N,) of float
        Bead interval widths expressed as segment lengths nondimensionalized
        by the Kuhn length of the polymer

    Returns
    -------
    np.ndarray (N,) of float
        Mean squared end-to-end distance of the internal segments corresponding
        to lengths in `seg_lengths_kuhn`, nondimensionalized by Kuhn length
        squared
    """
    return 2 * (
        seg_lengths_kuhn / 2 - (1 - np.exp(-2 * seg_lengths_kuhn)) / 2 ** 2
    )


def save_mean_r2(
    save_path: str, seg_lengths_kuhn: np.ndarray,
    avg_r2: np.ndarray, avg_r2_theory: np.ndarray
):
    """Save simulated and theoretical mean-squared end-to-end distances.

    Parameters
    ----------
    save_path : str
        Path at which to save results
    seg_lengths_kuhn : np.ndarray (N,) of float
        Bead interval widths expressed as segment lengths nondimensionalized
        by the Kuhn length of the polymer
    avg_r2 : np.ndarray (N,) of float
        Mean squared end-to-end distance of polymer segments with widths defined
        in bead_range, nondimensionalized by squared kuhn length
    avg_r2_theory : np.ndarray (N,) of float
        Mean squared end-to-end distance of the internal segments corresponding
        to lengths in `seg_lengths_kuhn`, nondimensionalized by Kuhn length
        squared
    """
    with open(save_path, "w") as output_file:
        output_file.write(f"segment_length_kuhn,mean_r2,mean_r2_theory\n")
        for i, val in enumerate(avg_r2):
            output_file.write(
                f"{seg_lengths_kuhn[i]},{val},{avg_r2_theory[i]}\n"
            )


def main(
    output_dir: str,
    sim_ind: int,
    num_equilibration: int,
    save_path: str,
    lp: float,
    bead_spacing: float,
    *,
    lower_log: Optional[float] = -1.0,
    upper_log: Optional[float] = 2.0,
    log_step: Optional[float] = 0.05,
    sim_prefix: Optional[str] = "sim_"
):
    """Save simulated and theoretical mean-squared end-to-end distances.
    
    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    num_equilibration : int
        Number of equilibration snapshots to exclude from snapshot paths
    save_path : str
        Path at which to save results
    lp : float
        Persistence length of the polymer
    bead_spacing : float
        Segment distance between adjacent beads of the polymer
    lower_log, upper_log : Optional[float]
        Lower and upper bounds of the bead interval widths on a log scale
    log_step : Optional[float]
        Spacing of the bead interval widths on a log scale
    sim_prefix : Optional[str] 
        Prefix of simulation output directory names (default = "sim_")
    """
    output_paths = get_snapshot_paths(
        output_dir, sim_ind, num_equilibration, sim_prefix=sim_prefix
    )
    bead_range = get_interval_widths(lower_log, upper_log, log_step)
    seg_length_kuhn = get_interval_widths_kuhn(bead_range, bead_spacing, lp)
    avg_r2 = get_mean_r2(output_paths, bead_range, lp)
    r2_theory = get_avg_r2_theory(seg_length_kuhn)
    save_mean_r2(save_path, seg_length_kuhn, avg_r2, r2_theory)

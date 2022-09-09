"""Generate radial density distribution from theory and simulation.

Notes
-----
This module may be used to generate the radial density distribution from
simulation outputs or from theory for a Rouse polymer. The model assumes that
a Rouse polymer is restricted by a spherical confinement. This module provides
utility functions, but is not intended to be run independently.

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 6, 2022
"""

# Built-in Modules
import os
import sys
from typing import List, Tuple, Optional

# Third Party Modules
import numpy as np
import pandas as pd

# Custom Modules
from analyses.characterizations.inspect_simulations import get_snapshot_paths


def get_radial_densities(
        output_paths: List[str], r_min: float, r_max: float,
        r_step: Optional[float] = 100.
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Get the radial density distribution.

    Parameters
    ----------
    output_paths : List[str]
        Paths to the polymer configuration outputs at equilibrated
        snapshots
    r_min, r_max : float
        Minimum and maximum radial positions for which to calculate density;
        minimum radial position is selected to avoid numerical instabilities
        at radial positions close to 0; maximum radial position is most often
        set equal to the radius of confinement
    r_step : Optional[float]
        Step size of radial distance for which to calculate densities
        (default = 100.)

    Returns
    -------
    num_beads : int
        Number of beads in density calculation
    bin_edges : np.ndarray (N,) of float
        Edges of bins for radial position corresponding to radial densities
    radial_densities : np.ndarray (N,) of float
        Bead densities in bins of radial positions
    """
    counts_all = []
    if len(output_paths) == 0:
        raise ValueError("No output paths provided.")
    bins = np.arange(r_min, r_max, r_step)
    for output_path in output_paths:
        r = pd.read_csv(
            output_path, usecols=[1, 2, 3], header=None, skiprows=[0, 1]
        ).dropna().to_numpy()
        radial_dists = np.linalg.norm(r, axis=1)
        counts, bin_edges = np.histogram(radial_dists, bins=bins)
        counts = counts.astype(float)
        counts_all.append(counts)
    counts_all = np.array(counts_all)
    radial_densities = np.sum(counts_all, axis=0)
    for i in range(len(bin_edges)-1):
        volume = 4/3 * np.pi * ((bin_edges[i+1])**3 - (bin_edges[i])**3)
        radial_densities[i] /= volume
    num_beads = len(r)
    radial_densities /= np.sum(radial_densities) * r_step
    return num_beads, bin_edges, radial_densities


def get_theoretical_radial_densities(
        confine_radius: float, lp: float, r_min: float, r_max: float,
        r_step: float, num_beads: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the theoretical radial density distribution.

    Notes
    -----
    Theoretical radial densities are calculated assuming that the polymer
    behaves as a Rouse chain.

    Parameters
    ----------
    confine_radius : float
        Confinement radius
    lp : float
        Persistence length of the polymer
    r_min, r_max : float
        Minimum and maximum radial distances for which to evaluate
        theoretical density
    r_step : float
        Step size in radial positions for which theoretical radial densities
        will be evaluated
    num_beads : int
        Number of beads in density calculation

    Returns
    -------
    r_theory : np.ndarray (N,) of float
        Edges of bins for radial position corresponding to radial densities
    radial_densities_theory : np.ndarray (N,) of float
        Theoretical bead densities in bins of radial positions
    """
    a = confine_radius
    b = lp
    N = num_beads
    r_theory = np.arange(r_min, r_max, r_step)
    n_max = 1000    # Numerical parameter (unlikely to change)
    rho = np.zeros(len(r_theory))
    for n in range(2, n_max + 1):
        rho += (-1)**(n+1) / (n * np.pi) * np.sin(np.pi * r_theory / a) * \
               np.sin(n * np.pi * r_theory / a) / (
                       r_theory**2 * b**2 * (n**2 - 1)
               )
    rho += N / np.pi * np.sin(np.pi * r_theory / a)**2 / r_theory**2
    radial_densities_theory = rho / np.sum(rho) * r_step
    return r_theory, radial_densities_theory

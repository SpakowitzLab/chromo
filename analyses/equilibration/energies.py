"""Calculate the configurational energy of a polymer in a field.

Usage:      python energies.py <OUTPUT_DIR> <INT_SIM_INDEX> <SAVE_FILE>

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 7, 2022

Notes
-----
As of the current implementation, this module can only compute
the configurational enregy of Chromatin chains. To generalize this
to different wormlike chains, add the persistence length as an
argument to functions that compute energy, change the polymer
model used to compute elastic energy, and add compatibility with
custom-defined binders.

"""

# Built-in Modules
import os
import sys
import json
from typing import Optional, Tuple, Dict

# Third-party Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom Modules
import chromo.polymers as poly
import chromo.fields as fld
import chromo.binders
import analyses.characterizations.inspect_simulations as inspect


def parse_configurational_output(
    df_snapshot: pd.DataFrame,
    get_avg_bead_length: Optional[bool] = False
) -> Dict[str, np.ndarray]:
    """Parse a output file to get the associated polymer configuration.

    Parameters
    ----------
    df_snapshot : pd.DataFrame
        Pandas data frame loaded from a configurational snapshot
    get_avg_bead_length : Optional[bool]
        Flag indicating whether or not to compute average bead length
        (default = False)
    
    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of configurational parameter to values for the polymer;
        values of each configurational parameter are expressed as
        (contiguous) numpy arrays where rows represent individual beads
    """
    snapshot = {}
    snapshot["r"] = np.ascontiguousarray(df_snapshot["r"].to_numpy())
    snapshot["t3"] = np.ascontiguousarray(df_snapshot["t3"].to_numpy())
    snapshot["t2"] = np.ascontiguousarray(df_snapshot["t2"].to_numpy())
    snapshot["states"] = np.ascontiguousarray(df_snapshot["states"].to_numpy())
    snapshot["chemical_mods"] = np.ascontiguousarray(
        df_snapshot["chemical_mods"].to_numpy()
    )
    if get_avg_bead_length:
        snapshot["bead_length"] = np.average(
            np.linalg.norm(np.diff(snapshot["r"], axis=0), axis=1)
        )
    return snapshot


def init_polymer_field(
    sim_name: str, sim_path: str, df_snapshot: pd.DataFrame
) -> Tuple[poly.Chromatin, fld.UniformDensityField]:
    """Initialize polymer and field objects from configurational snapshot.

    TODO: Constructing the field and binder objects are useful functions;
    consider separating those into different functions in the
    `analyzes.characterizations.inspect_simulations` module or as class
    methods in the `chromo.binders` and `chromo.fields` modules then
    importing them here.

    Parameters
    ----------
    sim_name : str
        Name of the simulation output directory
    sim_path : str
        Path to the simulation output directory
    df_snapshot : pd.DataFrame
        Pandas data frame loaded from a configurational snapshot

    Returns
    -------
    poly.Chromatin
        Chromatin object matching the configurational output
    fld.UniformDensityField
        Field object matching the configurational output
    """
    snapshot = parse_configurational_output(
        df_snapshot, get_avg_bead_length=True
    )
    binder_names = np.ascontiguousarray(
        list(df_snapshot["states"].columns)
    )
    chemical_mod_names = np.ascontiguousarray(
        list(df_snapshot["chemical_mods"].columns)
    )
    field_params = inspect.get_field_parameters(
        sim_names=[sim_name], sim_paths = {sim_name: sim_path}
    )[sim_name]
    binder_params = inspect.get_binder_params(
        sim_names=[sim_name], sim_paths = {sim_name: sim_path}
    )[sim_name]
    polymer = poly.Chromatin(
        "polymer", snapshot["r"], bead_length=snapshot["bead_length"],
        t3=snapshot["t3"], t2=snapshot["t2"], states=snapshot["states"],
        binder_names=binder_names, chemical_mods=snapshot["chemical_mods"],
        chemical_mod_names=chemical_mod_names
    )
    binder_objs = [
        chromo.binders.get_by_name(binder) for binder in binder_names
    ]
    for i, binder_name in enumerate(binder_names):
        for binder_prop, val in binder_params[binder_name].items():
            if isinstance(val, str):
                val = json.loads(val.replace("\'", "\""))
                val = {k: float(v) for k, v in val.items()}
            setattr(binder_objs[i], binder_prop, val)

    binders = chromo.binders.make_binder_collection(binder_objs)
    field = fld.UniformDensityField(
        [polymer], binders, field_params["x_width"],
        field_params["nx"], field_params["y_width"],
        field_params["ny"], field_params["z_width"],
        field_params["nz"], confine_type= field_params["confine_type"],
        confine_length=field_params["confine_length"],
        chi=field_params["chi"], vf_limit=field_params["vf_limit"],
        assume_fully_accessible=field_params["assume_fully_accessible"],
        fast_field=field_params["fast_field"]
    )
    return polymer, field


def get_configurational_energies(
    output_dir: str, sim_ind: int, polymer_prefix: Optional[str] = "Chr",
    sim_prefix: Optional[str] = "sim_",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the total configurational energy of a polymer in a field.

    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
        (default = "Chr")
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")

    Returns
    -------
    np.ndarray (N,) of float
        Array of polymer energy at each configurational snapshot over the
        course of the simulation
    np.ndarray (N,) of float
        Array of field energy at each configurational snapshot over the
        course of the simulation
    np.ndarray (N,) of float
        Array of total polymer + field energy at each configurational
        snapshot over the course of the simulation
    """
    snap_paths = inspect.get_snapshot_paths(
        output_dir, sim_ind, polymer_prefix=polymer_prefix,
        sim_prefix=sim_prefix, num_equilibration=0
    )
    sim_name = f"{sim_prefix}{sim_ind}"
    sim_path = f"{output_dir}/{sim_name}"
    initialized = False
    polymer_energies = []
    field_energies = []
    all_energies = []

    for i, snap_path in enumerate(snap_paths):
        output = pd.read_csv(snap_path, header=[0, 1], index_col=0)
        if not initialized:
            polymer, field = init_polymer_field(sim_name, sim_path, output)
            initialized = True
        else:
            snapshot = parse_configurational_output(output)
            polymer.r = snapshot['r'].copy()
            polymer.t3 = snapshot["t3"].copy()
            polymer.t2 = snapshot["t2"].copy()
            polymer.states = snapshot["states"].copy()
            polymer.chemical_mods = snapshot["chemical_mods"].copy()
        
        polymer_energy = polymer.compute_E()
        field_energy = field.compute_E(polymer)
        polymer_energies.append(polymer_energy)
        field_energies.append(field_energy)
        all_energies.append(polymer_energy + field_energy)
    
    return (
        np.array(polymer_energies), np.array(field_energies),
        np.array(all_energies)
    )


def filter_energies(
    E: np.ndarray, cutoff_E: Optional[float] = 1E90
) -> np.ndarray:
    """Filter energies by removing excessively high values.
    
    Notes
    -----
    To force move rejections when boundary conditions are violated
    or nucleosome volume fractions get too high, energies of 1E99
    kT are applied; these high energies create a pressure for the
    system to adopt constraint-complient configurations over the
    course of the Monte Carlo simulation. However, these high
    energies can obscure more meaningful energy changes in our plot
    of energy convergence. `cutoff_E` specifies a strict energy cutoff
    above which energies are replaced with NaN.
    
    Parameters
    ----------
    E : np.ndarray (N,) of float
        Array of energy values
    cutoff_E : float
        Maximum energy, above which values are removed (default = 1E90
        kT)
        
    Returns
    -------
    np.ndarray (M,) of float
        Array of energies below the cutoff values; values from `E`
        above the `cutoff_E` are replaced with NaN
    """
    E[E > cutoff_E] = np.nan
    return E


def plot_configurational_energies(
    output_dir: str, sim_ind: int, save_file: str,
    polymer_prefix: Optional[str] = "Chr",
    sim_prefix: Optional[str] = "sim_",
    filter_E: Optional[float] = 1E90
):
    """Plot the configurational energy of a polymer in a field.

    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    save_file : str
        File name at which to save profile of energies; saved to the
        simulation output directory
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
        (default = "Chr")
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
    filter_E : Optional[float]
         Maximum energy to include in the plot (default = 1E90 kT)
    """
    E_poly, E_field, E_tot = get_configurational_energies(
        output_dir, sim_ind, polymer_prefix, sim_prefix
    )
    E_poly = filter_energies(E_poly, filter_E)
    E_field = filter_energies(E_field, filter_E)
    E_tot = filter_energies(E_tot, filter_E)
    sorted_snaps = np.arange(len(E_poly))
    plt.figure()
    plt.plot(sorted_snaps, E_poly, color='b', label='Polymer')
    plt.plot(sorted_snaps, E_field, color='r', label='Field')
    plt.plot(sorted_snaps, E_tot, color='k', label='Total')    
    plt.xlabel("Snapshot number")
    plt.ylabel(r"Energy ($kT$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sim_prefix}{sim_ind}/{save_file}", dpi=600)
    plt.close()


def main(output_dir, sim_ind, save_file):
    """Plot configurational energies over the course of a simulation.
    
    Notes
    -----
    See documentation for `plot_configurational_energies`.
    """
    plot_configurational_energies(output_dir, sim_ind, save_file)


if __name__ == "__main__":
    
    output_dir = sys.argv[1]
    sim_ind = int(sys.argv[2])
    save_file = sys.argv[3]
    
    print("Evaluating and plotting configurational energies...")
    main(output_dir, sim_ind, save_file)

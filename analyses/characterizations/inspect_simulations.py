"""Parse the physical parameters of simulation outputs.

Usage:      `python inspect_simulations.py <OUTPUT_DIR>`

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 5, 2022
"""

import os
import sys
from pprint import pprint
import json
from typing import Optional, Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_sim_paths(
    output_dir: str, sim_prefix: Optional[str] = "sim_",
    ind_delim: Optional[str] = "_"
) -> Tuple[List[str], List[int], Dict[str, str]]:
    """List paths to simulation output directories in sorted order.

    Parameters
    ----------
    output_dir : str
        Path to folder containing all simulation output directories
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
    ind_delim : Optional[str]
        Delimiter in simulation output directories preceeding simulation
        index (default = "_")

    Returns
    -------
    List[str]
        Sorted list of simulation output directory names
    List[int]
        Sorted list of simulation indices
    Dict[str, str]
        Mapping of simulation names to output directory paths
    """
    sim_names = os.listdir(output_dir)
    sim_names = [sim for sim in sim_names if sim.startswith(sim_prefix)]
    sim_inds = [int(sim.split(ind_delim)[-1]) for sim in sim_names]
    sim_names = [sim for _, sim in sorted(zip(sim_inds, sim_names))]
    sim_inds = np.sort(sim_inds)
    sim_paths = {}
    for sim in sim_names:
        sim_paths[sim] = f"{output_dir}/{sim}"
    return sim_names, sim_inds, sim_paths


def get_snapshot_paths(
    output_dir: str, sim_ind: int, num_equilibration: int,
    polymer_prefix: Optional[str] = "Chr",
    sim_prefix: Optional[str] = "sim_"
) -> List[str]:
    """List paths to equilibrated configurational snapshots.

    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    num_equilibration : int
        Number of equilibration snapshots to exclude from snapshot paths
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
        (default="Chr")
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")

    Returns
    -------
    List[str]
        List of paths to equilibrated configurational snapshots
    """
    sim_dir = f"{output_dir}/{sim_prefix}{sim_ind}"
    output_files = os.listdir(sim_dir)
    output_files = [
        f for f in output_files if f.endswith(".csv")
        and f.startswith(polymer_prefix)
    ]
    snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
    sorted_snap = np.sort(np.array(snapshot))
    output_files = [f for _, f in sorted(zip(snapshot, output_files))]
    output_files = [
        output_files[i] for i in range(len(output_files))
        if sorted_snap[i] > num_equilibration - 1
    ]
    output_paths = [f"{sim_dir}/{f}" for f in output_files]
    return output_paths


def get_specific_snapshot_path(
    output_dir: str, sim_ind: int, snap_ind: int,
    polymer_prefix: Optional[str] = "Chr",
    sim_prefix: Optional[str] = "sim_"
) -> str:
    """Get path to specific snapshot.
    
    Parameters
    ----------
    output_dir : str
        Path to directory containing all simulation outputs
    sim_ind : int
        Integer identifier for the simulation of interest
    snap_ind : int
        Snapshot index for which path to configurational file is desired
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
        (default = "Chr")
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
        
    Returns
    -------
    str
        Path to the polymer configuration file of the desired snapshot
    """
    sim_dir = f"{output_dir}/{sim_prefix}{sim_ind}"
    output_files = os.listdir(sim_dir)
    output_files = [
        f for f in output_files if f.endswith(".csv")
        and f.startswith(polymer_prefix)
    ]
    snapshot = [int(f.split("-")[-1].split(".")[0]) for f in output_files]
    sorted_snap = np.sort(np.array(snapshot))
    output_files = [f for _, f in sorted(zip(snapshot, output_files))]
    output_files = [
        output_files[i] for i in range(len(output_files))
        if sorted_snap[i] == snap_ind
    ]
    output_paths = [f"{sim_dir}/{f}" for f in output_files]
    return output_paths


def get_binder_params(
    sim_names: List[str], sim_paths: Dict[str, str],
    binder_file: Optional[str] = "binders"
) -> Dict[str, Dict[str, Any]]:
    """Get physical parameters of binding components.

    Notes
    -----
    This function parses the binder parameter file produced by the
    `to_file()` method of the `ReaderProtein` class.

    Parameters
    ----------
    sim_names : List[str]
        Sorted list of simulation output directory names
    sim_paths : Dict[str, str]
        Mapping of simulation names to output directory paths
    binder_file : Optional[str]
        Name of file in all simulation output directories containing
        the physical parameters of the binding components, as produced
        by the `to_file()` method of the `ReaderProtein` class
        (default = "binders")

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of simulation output directory names to dictionaries
        with keys indicating physical parameters of binders and values
        indicating their settings in the simulation
    """
    binder_params = {}
    for sim in sim_names:
        binder_path = f"{sim_paths[sim]}/{binder_file}"
        df_binder = pd.read_csv(binder_path, index_col="name")
        df_binder.dropna(axis=1, inplace=True)
        df_binder = df_binder.round(3)
        df_binder = df_binder.iloc[:, 1:]
        df_binder["cross_talk_field_energy_prefactor"] = \
            df_binder["cross_talk_field_energy_prefactor"].apply(
                lambda string_dict: {
                    k: round(float(v), 3)
                    for k, v in json.loads(
                        string_dict.replace("\'", "\"")
                    ).items()
                }
            )
        binder_dict = df_binder.to_dict(orient="index")
        binder_params[sim] = binder_dict
    return binder_params


def match_two_mark_sims(
    sim_names: List[str], binder_params: Dict[str, Dict[str, Any]],
    mark_1: Optional[str] = "HP1", mark_2: Optional[str] = "PRC1"
) -> Dict[str, List[str]]:
    """Match all simulation names sharing the same binder parameters.

    Notes
    -----
    This function is useful for matching coarse-grained and refined
    simulations by binder parameters.

    Parameters
    ----------
    sim_names : List[str]
        Sorted list of simulation output directory names
    binder_params : Dict[str, Dict[str, Any]]
        Mapping of simulation output directory names to dictionaries
        with keys indicating physical parameters of bindersand values
        indicating their settings in the simulation
    mark_1, mark_2 : Optional[str]
        Names of the binders for which physical parameters are being
        matched between simulations (defaults are "HP1" and "PRC1",
        respectively)

    Returns
    -------
    Dict[str, List[str]]
        Mapping of binder physical parameter shorthands to lists of
        simulation names matching the physical parameter shorthands;
        the binder physical parameter shorthand is a semicolon-delimited
        string of the following:
            (1) mark_1 chemical potential
            (2) mark_2 chemical potential
            (3) mark_1-mark_1 interaction energy
            (4) mark_2-mark_2 interaction energy
            (5) mark_1-mark_2 interaction energy
    """
    matching_sims = {}
    for sim in sim_names:
        binder_dict = binder_params[sim]
        cp_mark_1 = round(binder_dict[mark_1]["chemical_potential"], 3)
        cp_mark_2 = round(binder_dict[mark_2]["chemical_potential"], 3)
        self_mark_1 = round(binder_dict[mark_1]["interaction_energy"], 3)
        self_mark_2 = round(binder_dict[mark_2]["interaction_energy"], 3)
        cross = round(
            json.loads(
                binder_dict[mark_1]["cross_talk_interaction_energy"].replace(
                    "\'", "\""
                )
            )[mark_2], 3
        )
        params = f"{cp_mark_1};{cp_mark_2};{self_mark_1};{self_mark_2};{cross}"
        if params not in matching_sims.keys():
            matching_sims[params] = [sim]
        else:
            matching_sims[params].append(sim)
    return matching_sims


def convert_to_numeric(init_val) -> Any:
    """Try to convert strings to numerical values.

    Notes
    -----
    If the string denotes a float-point value, it will
    be rounded to three decimal places
    """
    try:
        val = float(init_val)
        if val.is_integer():
            val = int(val)
        else:
            val = round(val, 3)
        return val
    except:
        return init_val


def get_field_parameters(
    sim_names : List[str], sim_paths = Dict[str, str],
    field_file : Optional[str] = "UniformDensityField"
) -> Dict[str, Dict[str, Any]]:
    """Get physical parameters of the field.

    Notes
    -----
    This function parses the field parameter file produced by the
    `to_file()` method of the `UniformDensityField` class.

    Parameters
    ----------
    sim_names : List[str]
        Sorted list of simulation output directory names
    sim_paths : Dict[str, str]
        Mapping of simulation names to output directory paths
    field_file : Optional[str]
        Name of file in all simulation output directories containing
        the physical parameters of the field, as produced by the
        `to_file()` method of the `UniformDensityField` class
        (default = "UniformDensityField")

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of simulation output directory names to dictionaries
        with keys indicating physical parameters of fields and values
        indicating their settings in the simulation
    """
    field_params = {}
    for sim in sim_names:
        field_path = f"{sim_paths[sim]}/{field_file}"
        df_field = pd.read_csv(
            field_path, header=None, names=("Label", "Value"),
            index_col=0
        )
        df_field["Value"] = df_field["Value"].apply(
            lambda val: convert_to_numeric(val)
        )
        field_dict = df_field.to_dict(orient="dict")["Value"]
        field_params[sim] = field_dict
    return field_params


def create_parameter_json(
        sim_names: List[str],
        binder_params: Dict[str, Dict[str, Any]],
        field_params: Dict[str, Dict[str, Any]]
) -> str:
    """Generate JSON formatted string from physical parameters.

    Notes
    -----
    Combines binder parameters and field parameters for simulations
    listed in `sim_names`, then outputs the combined parameters as
    a JSON formatted string.

    Parameters
    ----------
    sim_names : List[str]
        Sorted list of simulation output directory names
    binder_params : Dict[str, Dict[str, Any]]
        Mapping of simulation output directory names to dictionaries
        with keys indicating physical parameters of bindersand values
        indicating their settings in the simulation
    field_params : Dict[str, Dict[str, Any]]
        Mapping of simulation output directory names to dictionaries
        with keys indicating physical parameters of fields and values
        indicating their settings in the simulation

    Returns
    -------
    str
        JSON formatted string of binder parameters and field
        parameters by simulation output directory names
    """
    all_params = {}
    for sim in sim_names:
        param_dict = {
            "binder_parameters": binder_params[sim],
            "field_parameters": field_params[sim]
        }
        all_params[sim] = param_dict
    params_json = json.dumps(all_params, indent=4)
    return params_json


def main(output_dir: Optional[str] = "output"):
    """Print paired sims and physical parameters at the specified output dir.

    Parameters
    ----------
    output_dir : Optional[str]
        Output directory for which to parse physical parameters of all
        simulations (default = "output")
    """
    sim_names, sim_inds, sim_paths = get_sim_paths(output_dir)
    binder_params = get_binder_params(sim_names, sim_paths)
    matching_sims = match_two_mark_sims(sim_names, binder_params)

    print("Paired Simulations: ")
    pprint(matching_sims)
    print()

    field_params = get_field_parameters(sim_names, sim_paths)
    params_json = create_parameter_json(sim_names, binder_params, field_params)

    print("Physical Parameter JSON: ")
    print(params_json)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("USAGE: `python inspect_simulations.py <OUTPUT_DIR>`")
        raise ValueError(
            "This module accepts up to one command line argument."
        )

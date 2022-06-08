"""Store modification pattern for an output given the output directory.

Joseph Wakim
April 28, 2022
"""

import os
import sys

import numpy as np
import pandas as pd


# Insert package root to system path
cwd = os.getcwd()
parent_dir = cwd + "/../.."
sys.path.insert(1, parent_dir)
os.chdir(parent_dir)


def get_binding_and_modification_patterns(
    sim_output_dir: str,
    results_output_dir: str,
    init_config_file: str,
    mark: str,
    reader: str,
    binding_pattern_file: str,
    binder_spec_file: str
):
    """Get the binding and modification patterns for a given output directory.

    Notes
    -----
    Name the file containing the modification pattern by its chemical potential.

    Parameters
    ----------
    sim_output_dir : str
        The simulation output directory containing the modification states
    results_output_dir : str
        Results directory in which to save average modification and binding
        states (relative to root directory of package)
    init_config_file : str
        The file containing the initial configuration of the biopolymer
    mark : str
        Name of the chemical modification for which to save modification pattern
    reader : str
        Name of reader protein for which to save binding pattern
    binding_pattern_file : str
        The file containing the average binding pattern for the biopolymer
    binder_spec_file : str
        The file containing the binder specifications from which we obtain the
        chemical potential
    """
    check_results_outdir(results_output_dir)
    mu = get_chemical_potential(sim_output_dir, binder_spec_file, reader)
    modification_pattern_result = get_result_file_name(
        results_output_dir, mu, f"mod_state_{mark}"
    )
    binding_pattern_result = get_result_file_name(
        results_output_dir, mu, f"avg_binding_state_{reader}"
    )
    save_binding_pattern(
        sim_output_dir, binding_pattern_result, binding_pattern_file,
        results_output_dir
    )
    save_modification_pattern(
        sim_output_dir, modification_pattern_result, init_config_file, mark,
        results_output_dir
    )


def check_results_outdir(results_output_dir: str):
    """ Check if the results output directory exists and if not, make it.

    Parameters
    ----------
    results_output_dir : str
        Name of the results output directory
    """
    if not os.path.isdir(results_output_dir):
        os.mkdir(results_output_dir)


def get_chemical_potential(
    sim_out_dir: str, binder_spec_file: str, reader: str
) -> float:
    """Get the chemical potential of the simulation.

    Parameters
    ----------
    sim_out_dir : str
        Output directory of the simulation
    binder_spec_file : str
        Name of the binder specification file in the output directory
    reader : str
        Name of reader protein for which to save binding pattern

    Returns
    -------
    float
        Chemical potential of the binder
    """
    binder_specs = pd.read_csv(
        f"{sim_out_dir}/{binder_spec_file}", sep=",", header=0,
        index_col="name", usecols=["name", "chemical_potential"]
    )
    return binder_specs.loc[reader, "chemical_potential"]


def get_result_file_name(
    results_output_dir: str, mu: float, descriptor: str
) -> str:
    """Generate a unique file name for the resulting pattern assessed.

    Parameters
    ----------
    results_output_dir : str
        Name of the output directory into which we will save the result
    mu : float
        Chemical potential associated with the result being saved
    descriptor : str
        Brief description of result, which will serve as a prefix to the result
        file name.

    Returns
    -------
    str
        Result file name
    """
    existing_files = set(os.listdir(results_output_dir))
    trailing_zeros = 0
    mu_float = float(mu)
    mu_str = str(mu_float).replace("-", "m").replace(".", "d")
    while True:
        proposed_trial_name = f"{descriptor}_{mu_str}{'0' * trailing_zeros}.csv"
        if proposed_trial_name not in existing_files:
            return proposed_trial_name
        trailing_zeros += 1


def save_binding_pattern(
    sim_output_dir: str,
    binding_pattern_result_file: str,
    binding_pattern_file: str,
    result_output_dir: str
):
    """Save the binding pattern of the chromosome.

    Parameters
    ----------
    sim_output_dir : str
        Output directory of the simulation
    binding_pattern_result_file : str
        Filename for which to save binding pattern
    binding_pattern_file : str
        File containing average binding pattern
    result_output_dir : str
        Output directory into which to save processed modification and binding
        profiles
    """
    original_file = pd.read_csv(
        f"{sim_output_dir}/{binding_pattern_file}", header=None,
    )
    original_file.to_csv(
        f"{result_output_dir}/{binding_pattern_result_file}",
        header=False,
        index=False
    )


def save_modification_pattern(
    sim_output_dir: str,
    modification_pattern_result_file: str,
    init_config_file: str,
    mark: str,
    result_output_dir: str
):
    """

    Parameters
    ----------
    sim_output_dir : str
        Output directory of the simulation
    modification_pattern_result_file : str
        Filename for which to save modification pattern
    init_config_file : str
        File in output directory containing initial configuraiton
    mark : str
        Name of the chemical modification for which we wish to save modification
        pattern
    result_output_dir : str
        Output directory into which to save processed modification and binding
        profiles
    """
    init_config = pd.read_csv(
        f"{sim_output_dir}/{init_config_file}",
        header=[0, 1],
        sep=",",
        index_col=0
    )
    modifications = init_config.loc[:, ("chemical_mods", mark)]
    modifications.to_csv(
        f"{result_output_dir}/{modification_pattern_result_file}",
        header=None,
        index=False
    )


def main():
    """Generate the modification and binding profiles for each output directory.
    """
    output_dir = "output"
    sim_IDs = np.arange(14, 79, 1)
    results_output_dir = "binding_profiles"
    init_config_file = "Chr-1"
    mark = "H3K9me3"
    reader = "HP1"
    binding_pattern_file = "avg_binding_pattern.csv"
    binder_spec_file = "binders"
    for ID in sim_IDs:
        sim_output_dir = f"{output_dir}/sim_{int(ID)}"
        get_binding_and_modification_patterns(
            sim_output_dir, results_output_dir, init_config_file, mark, reader,
            binding_pattern_file, binder_spec_file
        )


if __name__ == "__main__":
    main()

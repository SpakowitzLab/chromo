"""Given output file, analyze patterns of reader protein binding states.

This module may be used as a command line tool. For details, enter:
`python analyze_binding_states.py -h`

Corresponding author:   Joseph Wakim
Affiliation:            Spakowitz Lab, Stanford University
Date:                   March 9, 2021
"""
from typing import Sequence, Tuple, Optional
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cwd = os.getcwd()
parent_dir = cwd + "/../.."
os.chdir(parent_dir)
sys.path.insert(1, parent_dir)


class CommandLineUsage():
    """Parse command line arguments, if run as a command line tool.

    References
    ----------
    https://stackoverflow.com/questions/40710719/optional-command-line-arguments
    """

    def __init__(self):
        """Initialize the command line parser.
        """
        parser = argparse.ArgumentParser(
            description="Analyze reader protein binding patterns."
        )
        parser.add_argument(
            "-s", "--sim",
            help="Example: `--sim 1` will load simulation snapshot 1.",
            type=int,
            metavar="",
            required=True
        )
        parser.add_argument(
            "-p", "--protein",
            help="Example: `--protein HP1` will load patterns of HP1 binding.",
            metavar="",
            required=True
        )
        parser.add_argument(
            "-l",
            "--lower",
            help="Example: `--lower 20` averages binding states starting at "
                 "snapshot 20.",
            metavar="",
            type=int,
            required=True
        )
        parser.add_argument(
            "-u",
            "--upper",
            help="Example: `--upper 100` averages binding states up to "
                 "snapshot 100.",
            required=False,
            type=int,
            metavar=""
        )
        parser.add_argument(
            "-d", "--output_dir",
            help="Example: `--output_dir output` specifies that simulation "
                 "outputs come from `output` directory, relative to root.",
            required=False,
            metavar="",
            default="output"
        )
        parser.add_argument(
            "-o", "--seq_output",
            help="Example: `--seq_output avg_hp1.csv` specifies that average "
                 "binding pattern will be saved into file `avg_hp1.csv`.",
            required=False,
            metavar="",
            default="avg_binding_pattern.csv"
        )
        self.args = parser.parse_args()


def get_file_list(
    sim_id: int,
    snapshot_range: Tuple[int, Optional[int]],
    output_dir: Optional[str] = "output"
) -> Sequence[str]:
    """Generate list of output file names for range of simulation snapshots.

    Notes
    -----
    File names include `.csv` extensions.

    Parameters
    ----------
    sim_id : int
        Identifer of simulation for which binding state is to be averaged
    snapshot_range : Tuple[int, Optional[int]]
        Bounds of snapshot numbers into which average binding state is to be
        calculated. In the form (LowerBound, UpperBound) or (LowerBound). If
        only one bound is provided in the tuple, then that bound will be
        interpreted as the lower bound, and the upper bound will be placed at
        the maximum snapshot available. If both bounds are provided, then the
        average binding state will be determined between the two values.
    output_dir : Optional[str]
        Path to output directory, relative to root directory (default =
        "output")

    Returns
    -------
    Sequence[str]
        List of file names for selected output snapshot configurations
    """
    sim_output_dir = f"{output_dir}/sim_{sim_id}/"
    all_content = os.listdir(sim_output_dir)
    snapshots = [
        file for file in all_content if
        (file.endswith(".csv") and file.startswith("Chr"))
    ]
    if len(snapshot_range) == 1 or snapshot_range[1] is None:
        file_list = [
            snapshot for snapshot in snapshots if
            int(snapshot.split(".")[0].split("-")[-1]) >= snapshot_range[0]
        ]
    elif (len(snapshot_range) == 2 and snapshot_range[0] is not None and
          snapshot_range[0] is not None):
        file_list = [
            snapshot for snapshot in snapshots if
            (snapshot_range[0] <= int(snapshot.split(".")[0].split("-")[-1]) <=
             snapshot_range[1])
        ]
    else:
        raise ValueError("Dimensions of `snapshot_range` is invalid.")
    return file_list


def get_binding_seq(file_path: str, protein_name: str) -> np.ndarray:
    """Get the binding sequence for a specified reader protein.

    Parameters
    ----------
    file_path : str
        Path to the snapshot configuration output file
    protein_name : str
        Name of the reader protein for which sequence is to be retrieved

    Returns
    -------
    array_like (N,) of int
        Pattern of reader protein binding states in specified output
        configuration
    """
    return pd.read_csv(
        file_path, header=[0, 1], index_col=0
    ).loc[:, (["states", "chemical_mods"], protein_name)].to_numpy()


def get_avg_binding_state(
    sim_id: int,
    snapshot_range: Tuple[int, Optional[int]],
    protein_name: str,
    output_dir: Optional[str] = "output",
    save_name: Optional[str] = None
) -> np.ndarray:
    """Get the average reader protein binding state across snapshots.

    Parameters
    ----------
    sim_id : int
        Identifer of simulation for which binding state is to be averaged
    snapshot_range : Tuple[int, Optional[int]]
        Bounds of snapshot numbers into which average binding state is to be
        calculated. In the form (LowerBound, UpperBound) or (LowerBound). If
        only one bound is provided in the tuple, then that bound will be
        interpreted as the lower bound, and the upper bound will be placed at
        the maximum snapshot available. If both bounds are provided, then the
        average binding state will be determined between the two values.
    protein_name : str
        Name of the protein for which average binding states are to be
        calculated.
    output_dir : Optional[str]
        Path to output directory, relative to root directory (default =
        "output")
    save_name : Optional[str]
        File name with which to save average pattern of reader protein binding
        state (default = None). Do not include extension. If None, then no
        output file will be saved.

    Returns
    -------
    array_like (N,) of double
        Array of average binding state at each monomer for specified reader
        protein.
    """
    file_list = get_file_list(sim_id, snapshot_range, output_dir)
    n_files = len(file_list)

    for i, file in enumerate(file_list):
        file_path = f"{output_dir}/sim_{sim_id}/{file}"
        if i == 0:
            tot_binding_states = get_binding_seq(file_path, protein_name)
        else:
            tot_binding_states += get_binding_seq(file_path, protein_name)

    avg_binding_states = tot_binding_states / n_files

    if save_name is not None:
        np.savetxt(
            f"{output_dir}/sim_{sim_id}/{save_name}.csv",
            avg_binding_states,
            delimiter=","
        )

    return avg_binding_states


def get_sliding_window_avg(seq: np.ndarray, window_size: int) -> np.ndarray:
    """Get the sliding window average protein binding state.

    References
    ----------
    https://www.delftstack.com/howto/python/moving-average-python/

    Parameters
    ----------
    seq : array_like (N,) of double
        Sequence of protein binding states
    window_size : int
        Window size for sliding window average binding state

    Returns
    -------
    array_like (N - window_size + 1,) of double
        Window-averaged reader protein binding state
    """
    return np.convolve(seq, np.ones(window_size), mode='valid') / window_size


def plot_avg_binding_state(seq: np.ndarray, save_path: str):
    """Plot a barchart of the average reader protein binding state.

    Parameters
    ----------
    seq : array_like (N,) of double
        Sequence of average reader protein binding states
    save_path : str
        Path at which to save barchart of average HP1 binding state
    """
    ind = np.arange(0, len(seq), 1)
    plt.figure(figsize=(15, 2))
    plt.bar(ind, seq.flatten())
    plt.xlabel("Nucleosome Index")
    plt.ylabel("Avg. Bind")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


def main(
    sim_id: int,
    snapshot_range: Tuple[int, Optional[int]],
    protein_name: str,
    output_dir: Optional[str] = "output",
    save_name: Optional[str] = None
):
    """Save the average reader protein binding state between specified bounds.

    Parameters
    ----------
    sim_id : int
        Identifer of simulation for which binding state is to be averaged
    snapshot_range : Tuple[int, Optional[int]]
        Bounds of snapshot numbers into which average binding state is to be
        calculated. In the form (LowerBound, UpperBound) or (LowerBound). If
        only one bound is provided in the tuple, then that bound will be
        interpreted as the lower bound, and the upper bound will be placed at
        the maximum snapshot available. If both bounds are provided, then the
        average binding state will be determined between the two values.
    protein_name : str
        Name of the protein for which average binding states are to be
        calculated.
    output_dir : Optional[str]
        Path to output directory, relative to root directory (default =
        "output")
    save_name : Optional[str]
        File in which to save average pattern of reader protein binding state
        (default = None). If None, then no output file will be saved.
    """
    seq = get_avg_binding_state(
        sim_id, snapshot_range, protein_name, output_dir, save_name
    )
    save_path = f"{output_dir}/sim_{sim_id}/{save_name}.png"
    plot_avg_binding_state(seq, save_path)


if __name__ == "__main__":
    cl = CommandLineUsage()
    sim_id = cl.args.sim
    protein = cl.args.protein
    bounds = (cl.args.lower, cl.args.upper)
    output_dir = cl.args.output_dir
    save_out = cl.args.seq_output
    main(sim_id, bounds, protein, output_dir, save_out)

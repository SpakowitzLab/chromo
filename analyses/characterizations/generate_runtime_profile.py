"""List the top functions by cumulative runtime.

Usage:      `python generate_runtime_profile.py <CPROFILE_OUTPUT_PATH> <SAVE_PATH>`

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 5, 2022

Notes
-----
Before running this script, run the cProfile on the script being analyzed.
To run cProfile on a script, use:

'python -m cProfile -s "cumulative" -o <CPROFILE_OUTPUT_NAME> <PY_SIM_RUN_FILE>'
"""

import sys

import pstats
from pstats import SortKey


def main(
    cprofile_output_path: str, save_path: str
):
    """Analyze cProfile output to list top functions by cumulative runtime.
    
    Notes
    -----
    By default, cProfile outputs runtime profiles to the root directory of the
    package unless a subdirectory is specified. In this function, all paths
    must be specified relative to the directory containing this module. Use the 
    relative path prefix "../.." to navigate to the root directory of this
    package, if this is where the cProfile output is.
    
    Parameters
    ----------
    cprofile_output_path : str
        Path to the cProfile output (relative to cwd)
    save_path : str
        Path (relative to cwd) at which to save list of functions ordered by
        cumulative runtime
    """
    with open(save_path, "w") as f:
        p = pstats.Stats(cprofile_output_path, stream=f)
        p.strip_dirs()
        p.sort_stats(SortKey.CUMULATIVE).print_stats(500)
    

if __name__ == "__main__":
    cprofile_output_path = sys.argv[1]
    save_path = sys.argv[2]
    main(cprofile_output_path, save_path)

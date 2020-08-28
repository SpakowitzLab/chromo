"""
A module with routines designed to make our code "automatically" reproducible.

Reproducible here should be taken in the scientific sense. We want to, given
the output of our simulation, always be able to
    1. Reconstruct the inputs to the simulation exactly.
    2. Run our code again with those inputs to generate that exact same output.

For our purposes, this means saving
    1. The Git commit hash of the simulation used.
    2. All random seeds used.
    3. Serializing all inputs used.

If in doubt, you probably want to simply wrap any simulation function you write
with :ref:`make_reproducible`.
"""
from pathlib import Path
import inspect

import pandas as pd

# get version number (includes commit hash) from versioneer
from .._version import get_versions

__version__ = get_versions()['version']
del get_versions

# enforce a uniform kwarg for output directory
output_dir_param_name = 'output_dir'
# output_dir_param_name / sim_tracker_name will store history of simulations
# run in that output directory
sim_tracker_name = Path('simulations.csv')
# the actual simulation output will go in
sim_folder_prefix = Path('sim_')

def make_reproducible(sim):
    f"""
    Decorator for automagically making simulations reproducible.

    Requires every input to the function to have a __repr__ that is actually
    useful, and that the wrapped function has a keyword argument with name
    {output_dir_param_name}. Also requires you to pass a "random_seed" kwarg.

    The versioneer version (including a commit hash if necessary), the
    date/time, a unique folder name where the simulation inputs and output will
    be stored, and any arguments to the function with built-in types will be
    will be appended as a row to the {sim_tracker_name} file, for easy opening
    as a Pandas DataFrame tracking all simulations that have been done in a
    given directory.

    The unique output directory name will be forwarded into the simulation
    function via the {output_dir_param_name} kwarg, meaning that simulation
    output will typically be in a subfolder of the requested folder to prevent
    overwriting old output.

    The final directory structure, if you request

        >>> {output_dir_param_name} = "output_dir"

    at run time, will look like::

        output_dir
        ├── {sim_tracker_name}
        ├── {sim_folder_prefix}1
        │   ├── param1
        │   ├── ...
        │   ├── paramN
        │   └── (Any output of simulator goes here)
        ├── ...
        └── {sim_folder_prefix}M
            ├── param1
            ├── ...
            ├── paramN
            └── (Any output of simulator goes here)
    """
    params = inspect.signature(sim)
    if "random_seed" not in params:
        raise ValueError("Requires the random seed to be passed in explicitly,"
                         " even if the random generator is passed to the "
                         "function as well.")
    if output_dir_param_name not in params:
        raise ValueError(f"Could not wrap simulation function: {sim}\n"
                         f"Missing required kwarg: {output_dir_param_name}.")
    outdir_param = params[output_dir_param_name]
    def wrapper(*args, **kwargs):
        if output_dir_param_name not in kwargs:
            if outdir_param.default is inspect.Parameter.empty:
                raise ValueError(f"Could not run simulation function: {sim}\n"
                                 f"{output_dir_param_name} not specified, and "
                                 "no default value can be found.")
            kwargs[output_dir_param_name] = outdir_param.default
        # get a new folder to save simulation into in a thread-safe way
        outdir = kwargs[output_dir_param_name]
        output_subfolder = get_unique_subfolder(Path(outdir)/sim_folder_prefix)
        kwargs[output_dir_param_name] = output_subfolder
        # get the inputs that can be simply saved in our CSV file
        simple_params, hard_params = split_builtin_params(sim, *args, **kwargs)
        # append to the CSV file, unless it doesn't exist
        # TODO make thread-safe
        sim_tracker = output_subfolder / sim_tracker_name
        header = not sim_tracker.exists()
        simple_params = pd.DataFrame(pd.Series(simple_params)).T
        simple_params.to_csv(sim_tracker, header=header, index=False, mode='a')
        # TODO add support for non-simple params
        return sim(*args, **kwargs)
    return wrapper


# the following strategy would also work, but would lead to us having to pickle
# some builtin things, like "map", "Error", etc..
# builtin_types = tuple(getattr(builtins, t) for t in dir(builtins)
#                       if isinstance(getattr(builtins, t), type))
builtin_types = [bool, bytes, bytearray, complex, float, int, str]
# the additional types are not allowed by default because they make loading in
# the CSV a little trickier
# ..., frozenset, tuple]


def split_builtin_params(sim, *args, **kwargs):
    sig = inspect.signature(sim).bind(*args, **kwargs)
    builtin_args = {}
    non_builtin = {}
    for arg_name, value in sig.arguments.items():
        if type(value) in builtin_types:
            builtin_args[arg_name] = value
        else:
            non_builtin[arg_name] = value
    return builtin_args, non_builtin


def get_unique_subfolder(root):
    """
    Create and return a thread-safe unique folder.

    Parameters
    ----------
    root : str or Path
        The base of the filename which we should try appending numbers to until
        we find a number which doesn't already exist.

    Returns
    -------
    Path
        The folder which was created.

    Notes
    -----
    Uses the fact that mkdir is required to be atomic on POSIX-compliant system
    to make sure that two threads aren't given the same folder.
    """
    i = 1
    while True:
        folder_name = Path(str(root) + str(i))
        try:
            folder_name.mkdir(parents=True)
            break
        except:
            i += 1
    return folder_name

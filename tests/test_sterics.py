"""Test Nucleosome Sterics.

These tests are designed to ensure that the identification of steric clashes
between nucleosomes is functioning as expected.
"""

import os
import sys
import time

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
parent_dir = cwd + "/.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
import chromo.polymers as poly

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())


def test_nucleosome_sterics():
    """Test the identification of steric clashes between nucleosomes.
    """

    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 10
    linker_lengths = np.array([linker_length] * (n_beads-1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Initialize the polymer
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Get the distances between pairs of nucleosomes
    p.get_distances()
    distances = p.distances.copy()
    distances_trial = p.distances_trial.copy()
    # Trial configuration should be equivalent to the initial configuration
    assert np.allclose(distances, distances_trial), \
        "Trial configuration should be equivalent to the initial configuration."

    # Verify that there are no steric clashes in the initial configuration
    n_clashes = p.check_steric_clashes(distances)
    assert n_clashes == 0, \
        "There should be no steric clashes in the initial configuration."
    n_clashes_trial = p.check_steric_clashes(distances_trial)
    assert n_clashes_trial == 0, \
        "There should be no steric clashes in the trial configuration."

    # Move a nucleosome to create a steric clash
    p.r[0, :] = p.r[1, :] + np.array([0.5, 0, 0])
    p.get_distances()
    distances = p.distances.copy()
    n_clashes = p.check_steric_clashes(distances)
    assert n_clashes == 1, \
        f"There should be 1 steric clash in the current configuration;" \
        f" there are {n_clashes}."
    n_clashes_trial = p.check_steric_clashes(distances_trial)
    assert n_clashes_trial == 0, \
        "There should be no steric clashes in the trial configuration."


def test_distance_runtime():
    """Get the average runtime for computing pairwise distances.

    We will check how long it takes to compute pairwise distances for a
    chromatin fiber with 100 nucleosomes.
    """
    # Define chromatin to match theory
    linker_length_bp = 36
    length_bp = 0.332
    linker_length = linker_length_bp * length_bp
    n_beads = 100
    linker_lengths = np.array([linker_length] * (n_beads-1))
    lp = 50.
    lt = 100.
    bp_wrap = 147.

    # Initialize the polymer
    p = poly.DetailedChromatinWithSterics.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Determine average time to get pairwise distances
    n_trials = 100
    times = np.zeros(n_trials)
    for i in range(n_trials):
        start = time.time()
        p.get_distances()
        times[i] = time.time() - start

    print(f"Average time to evaluate get_distances(): {np.mean(times)} s")

    # Raise an error if the runtime is too long. This is not a strict test,
    # but it is useful to know if the code has become too inefficient.
    assert np.mean(times) < 0.01, \
        f"The average time to evaluate get_distances() has become too large; " \
        f"the average runtime is {np.mean(times)} s. Please reduce this time " \
        f"to less than 0.01 s."


if __name__ == "__main__":
    test_nucleosome_sterics()
    test_distance_runtime()
    print("All tests passed.")

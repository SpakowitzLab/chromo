"""Linker lengths appear to increase with twist. Why is this?
"""

import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
parent_dir = cwd + "/.."
sys.path.insert(1, parent_dir)

print("Directory containing the notebook:")
print(cwd)

import numpy as np
import chromo.mc as mc
import chromo.polymers as poly
import chromo.binders
from chromo.fields import UniformDensityField
import chromo.mc.mc_controller as ctrl
from chromo.util.reproducibility import get_unique_subfolder_name
from wlcstat.chromo import gen_chromo_conf

os.chdir(parent_dir)
print("Root Directory of Package: ")
print(os.getcwd())


def test_dE_with_twist():
    """Stretching the polymer should not affect twist energy.
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

    # Instantiate the null reader protein
    # This is pre-defined in the `chromo.binders` module
    null_binder = chromo.binders.get_by_name('null_reader')

    # Create the binder collection
    binders = chromo.binders.make_binder_collection([null_binder])

    # Initialize the polymer
    p = poly.DetailedChromatin.straight_line_in_x(
        "Chr",
        linker_lengths,
        bp_wrap=bp_wrap,
        lp=lp,
        lt=lt,
        binder_names=np.array(["null_reader"])
    )

    # Compute the energy of the initial polymer configuration
    E_0 = p.compute_E()

    # Twist a bead by 2pi radians. Verify that the energy does not change.
    rot_matrix = np.array([[np.cos(2*np.pi), -np.sin(2*np.pi), 0],
                            [np.sin(2*np.pi), np.cos(2*np.pi), 0],
                            [0, 0, 1]])
    for i in range(n_beads):
        p.t3[i, :] = np.dot(rot_matrix, p.t3[i, :])
        p.t2[i, :] = np.dot(rot_matrix, p.t2[i, :])
        E_1 = p.compute_E()
        assert np.isclose(E_0, E_1), \
            "Twisting a bead by 2pi should not change the energy."

    # Stretching the polymer should not affect the twist energy.
    # Verify that the energy does not change.
    E_twist_0 = E_0 - p.compute_E_no_twist()
    for i in range(n_beads):
        p.r[i, :] = p.r[i, :] - np.array([1, 0, 0])
        E_2 = p.compute_E()
        E_twist_1 = E_2 - p.compute_E_no_twist()
        assert np.isclose(E_twist_0, E_twist_1), \
            "Stretching the polymer should not affect the twist energy."

    # When initialized as a straight chain, most energy should be in twist
    E_0 = p.compute_E()
    E_twist_0 = E_0 - p.compute_E_no_twist()
    frac_twist = E_twist_0 / E_0
    assert frac_twist > 0.9, \
        "When initialized as a straight chain, most energy should be in twist."

    # When initialized with the chain growth algorithm, a smaller fraction of
    # the energy should be in twist
    linker_lengths_bp = np.ones(n_beads-1, dtype=int) * linker_length_bp
    _, _, _, rn, un, orientations = gen_chromo_conf(
        linker_lengths_bp, lp=lp/length_bp, lt=lt/length_bp,
        return_orientations=True
    )
    p.r = rn.copy()
    p.r_trial = rn.copy()
    p.t3 = orientations["t3_incoming"].copy()
    p.t3_trial = orientations["t3_incoming"].copy()
    p.t2 = orientations["t2_incoming"].copy()
    p.t2_trial = orientations["t2_incoming"].copy()
    assert np.all(np.asarray(p.eps_twist) > 0), \
        "Twist modulus (eps_twist) should be positive."
    E_0 = p.compute_E()
    E_twist_0 = E_0 - p.compute_E_no_twist()
    assert E_twist_0 >= 0, \
        "Twist should not contribute negative energy."
    frac_twist_2 = E_twist_0 / E_0
    assert frac_twist_2 < frac_twist, \
        "When initialized with the chain growth algorithm, a smaller fraction" \
        " of the energy should be in twist."

    # Shifting the position of a bead should not affect the twist energy.
    for i in range(n_beads):
        p.r[i, :] = p.r[i, :] + np.array([i+1, 0, 0])
        E_3 = p.compute_E()
        assert not np.isclose(E_0, E_3), \
            "Shifting the position of a bead should affect the total energy."
        E_twist_3 = E_3 - p.compute_E_no_twist()
        assert np.isclose(E_twist_0, E_twist_3), \
            "Shifting the position of a bead should not affect the twist " \
            "energy."


if __name__ == "__main__":
    test_dE_with_twist()
    print("All tests passed!")

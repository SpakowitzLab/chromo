"""
Polymer class.

Creates a polymer with a defined length and discretization.
"""
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from abc import ABC

import numpy as np
import pandas as pd
from .util.init_cond import init_cond
from .util.find_parameters import find_parameters


@dataclass
class Epigenmark:
    """Information about the chemical properties of an epigenetic mark."""

    bind_energy: float
    interaction_energy: float
    chemical_potential: float


class State(Enum, ABC):
    """Defines the epigenetic state of a given bead for a single mark."""

    pass


class HP1_State(State):
    """HP1 can be bound to none, either, or both nucleosome tails."""

    NONE = 0
    LBOUND = 1
    RBOUND = 2
    BOTH = 3


class Polymer:
    """
    The positions and chemical state of a discrete polymer.

    The polymer carries around a set of coordinates ``r`` of shape
    ``(num_beads, 3)``, a triad of material normals ``t_i`` for ``i`` in
    ``{1,2,3}``, and some number of chemical states per bead.

    Its material properties are completely defined by these positions, chemical
    states, and the length of polymer (in Kuhn lengths) simulated by each bead.
    TODO: allow differential discretization, and decay to constant
    discretization naturally.

    Since this codebase is primarily used to simulate DNA, information about
    the chemical properties of each epigenetic mark are stored in `Epigenmark`
    objects.
    """

    def __init__(self, r, t_3, t_2, epigenmarks, states, length_per_bead):
        """
        Parameters
        ----------
        r : (N, 3) array_like of float
            The positions of each bead.
        t_3 : (N, 3) array_like of float
            The tangent vector to each bead in global coordinates.
        t_2 : (N, 3) array_like of float
            A material normal to each bead in global coordinates.
        epigenmarks : (M, ) Sequence[Epigenmark]
            Information about the chemical properties of each of the epigenetic
            marks being tracked.
        states : (N, M) array_like of int
            State of each of the M epigenetic marks being tracked for each
            bead.
        length_per_bead : float
            How many Kuhn lengths of polymer are simulated by each bead.
        """
        self.r = r
        self.t_3 = t_3
        self.t_2 = t_2
        self.epigenmarks = epigenmarks
        self.states = states
        self.length_per_bead = length_per_bead

    def from_files(*, r_file=None, t3_file=None, t2_file=None, mark_file=None,
                   states_file=None, lengths_file=None):
        """
        Instantiate a Polymer from input files.

        Each of the r, t_3, t_2, epigenmarks, states, and length_per_bead
        arrays requires a separate input file.
        """
        r = pd.read_csv(r_file, header=None) if r_file is not None else None
        t_3 = pd.read_csv(t3_file, header=None) if t3_file is not None else None
        t_2 = pd.read_csv(t2_file, header=None) if t2_file is not None else None
        states = pd.read_csv(states_file, header=None) if states_file is not None else None

        # Initialize the conformation, protein binding state, and the sequence of epigenetic marks
        self.r_poly, self.t3_poly, self.epigen_bind = init_cond(length_bead, 1, self.num_beads,
                                                                num_epigenmark, from_file, self.input_dir)

        # Load the polymer parameters for the conformational energy
        self.sim_type, self.eps_bend, self.eps_par, self.eps_perp, self.gamma, self.eta = find_parameters(length_bead)

        self.sequence = np.zeros((self.num_beads, num_epigenmark), 'd')
        for epigenmark_count in range(1, num_epigenmark + 1):
            seq_file = self.input_dir / Path("chromo" + str(self.polymer_count) + "seq" + str(epigenmark_count))
            if seq_file.exists():
                self.sequence[:, epigenmark_count - 1] = np.loadtxt(seq_file, delimiter=',')
            else:
                print(f"Sequence file does not exist for chromosome {self.polymer_count}")
                exit()

    def __str__(self):
        return f"{self.name} is a polymer with {self.num_beads} beads (each bead with " \
               f"{self.num_nucleo_per_bead} nucleosomes)"




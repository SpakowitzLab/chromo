"""
Polymer class.

Creates a polymer with a defined length and discretization.
"""
from pathlib import Path
from enum import Enum
import dataclasses

import numpy as np
import pandas as pd
from .util.find_parameters import find_parameters


@dataclasses.dataclass
class Epigenmark:
    """Information about the chemical properties of an epigenetic mark."""

    name: str
    bind_energy: float
    interaction_energy: float
    chemical_potential: float


class State:
    """Defines the epigenetic state of a given bead for a single mark."""

    pass


class HP1_State(State, Enum):
    """HP1 can be bound to none, either, or both nucleosome tails."""

    NONE = 0
    LBOUND = 1
    RBOUND = 2
    BOTH = 3


def make_epi_collection(epigenmarks):
    """
    Takes a sequence of epigenetic marks and returns a summary DataFrame.

    Parameters
    ----------
    epigenmarks : Sequence[Epigenmark]
        The epigenetic marks to be summarized.

    Returns
    -------
    pd.DataFrame
        Columns are the properties of Epigenmark.
    """
    df = pd.DataFrame(columns=['name', 'bind_energy', 'interaction_energy',
                               'chemical_potential'])
    for mark in epigenmarks:
        df = df.append(dataclasses.asdict(mark), ignore_index=True)
    return df


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

    def __init__(self, name, r, t_3, t_2, epigenmarks, states,
                 length_per_bead):
        """
        Parameters
        ----------
        name : str
            A name for convenient repr. Should be a valid filename.
        r : (N, 3) array_like of float
            The positions of each bead.
        t_3 : (N, 3) array_like of float
            The tangent vector to each bead in global coordinates.
        t_2 : (N, 3) array_like of float
            A material normal to each bead in global coordinates.
        epigenmarks : (M, ) Sequence[Epigenmark]
            Information about the chemical properties of each of the epigenetic
            marks being tracked.
        states : (N, M) array_like of State
            State of each of the M epigenetic marks being tracked for each
            bead.
        length_per_bead : float
            How many Kuhn lengths of polymer are simulated by each bead.
        """
        self.name = name
        self.r = r
        self.t_3 = t_3
        self.t_2 = t_2
        self.epigenmarks = make_epi_collection(epigenmarks)
        self.states = states
        self.length_per_bead = length_per_bead
        # Load the polymer parameters for the conformational energy
        self.sim_type, self.eps_bend, self.eps_par, self.eps_perp, \
            self.gamma, self.eta = find_parameters(length_per_bead)

    @classmethod
    def from_files(cls, *, r=None, t_3=None, t_2=None, marks=None, states=None,
                   lengths=None):
        """
        Instantiate a Polymer from input files.

        Each of the r, t_3, t_2, epigenmarks, states, and length_per_bead
        arrays requires a separate input file.
        """
        r = pd.read_csv(r, header=None) if r is not None else None
        t_3 = pd.read_csv(t_3, header=None) if t_3 is not None else None
        t_2 = pd.read_csv(t_2, header=None) if t_2 is not None else None
        states = pd.read_csv(states) if states is not None else None
        marks = pd.read_csv(r, header=None) if r is not None else None
        return cls(r, t_3, t_2, marks, states, lengths)

    @classmethod
    def straight_line_in_x(cls, num_beads, length_per_bead):
        r = np.zeros((num_beads, 3))
        r[:, 0] = length_per_bead * np.arange(num_beads)
        t_3 = np.zeros((num_beads, 3))
        t_3[:, 0] = 1
        t_2 = np.zeros((num_beads, 3))
        t_2[:, 1] = 1
        marks = None
        states = None
        return cls(r, t_3, t_2, marks, states, length_per_bead)

    def write_r(self, prefix, **kwargs):
        r_file = Path(str(prefix) + self.name + '_r.csv')
        np.savetxt(r_file, self.r, delimiter=',', **kwargs)

    def write_t2(self, prefix, **kwargs):
        t2_file = Path(str(prefix) + self.name + '_t2.csv')
        np.savetxt(t2_file, self.t2, delimiter=',', **kwargs)

    def write_t3(self, prefix, **kwargs):
        t3_file = Path(str(prefix) + self.name + '_t3.csv')
        np.savetxt(t3_file, self.t3, delimiter=',', **kwargs)

    def write_states(self, prefix, **kwargs):
        states_file = Path(str(prefix) + self.name + '_states.csv')
        np.savetxt(states_file, self.states, delimiter=',', **kwargs)

    def write_marks(self, prefix, **kwargs):
        marks_file = Path(str(prefix) + self.name + '_marks.csv')
        self.epigenmarks.to_csv(marks_file)

    @property
    def num_marks(self):
        return len(self.epigenmarks)

    def __str__(self):
        return f"Polymer<{self.name}, nbeads={self.num_beads}, " \
               f"nmarks={self.num_marks}>"

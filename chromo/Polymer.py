"""
Polymer class.

Creates a polymer with a defined length and discretization.
"""
from pathlib import Path
from enum import Enum
import dataclasses
import io

import numpy as np
import pandas as pd
from .util import dss_params, combine_repeat


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
            self.gamma, self.eta = self._find_parameters(length_per_bead)

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
    def straight_line_in_x(cls, name, marks, states, num_beads,
                           length_per_bead):
        r = np.zeros((num_beads, 3))
        r[:, 0] = length_per_bead * np.arange(num_beads)
        t_3 = np.zeros((num_beads, 3))
        t_3[:, 0] = 1
        t_2 = np.zeros((num_beads, 3))
        t_2[:, 1] = 1
        return cls(name, r, t_3, t_2, marks, states, length_per_bead)

    def write_marks(self, marks_file, **kwargs):
        self.epigenmarks.to_csv(marks_file, **kwargs)

    def __repr__(self):
        s = io.Bytes()
        self.write_repr(s)
        return s.decode()

    def write_repr(self, path):
        np.savez(path, r=self.r, t_2=self.t_2, t_3=self.t_3,
                 states=self.states, marks=self.epigenmarks.values,
                 length_per_bead=self.length_per_bead)

    @classmethod
    def from_repr(cls, path):
        npz = np.load(path)
        mark_info_arr = npz['marks']
        marks = [Epigenmark(*info) for info in mark_info_arr]
        return cls(r=npz['r'], t_3=npz['t_3'], t_2=npz['t_2'],
                   states=npz['states'], marks=marks,
                   length_per_bead=npz['length_per_bead'])

    @property
    def num_marks(self):
        return len(self.epigenmarks)

    @property
    def num_beads(self):
        return self.r.shape[0]

    def __str__(self):
        return f"Polymer<{self.name}, nbeads={self.num_beads}, " \
               f"nmarks={self.num_marks}>"

    @staticmethod
    def _find_parameters(length_bead):
        """Determine the parameters for the elastic forces based on ssWLC with twist."""
        sim_type = "sswlc"
        lp_dna = 53     # Persistence length of DNA in nm
        length_dim = length_bead * 0.34 / lp_dna        # Non-dimensionalized length per bead

        # Determine the parameter values using linear interpolation of the parameter table
        eps_bend = np.interp(length_dim, dss_params[:, 0], dss_params[:, 1]) / length_dim
        gamma = np.interp(length_dim, dss_params[:, 0], dss_params[:, 2]) * length_dim * lp_dna
        eps_par = np.interp(length_dim, dss_params[:, 0], dss_params[:, 3]) / (length_dim * lp_dna**2)
        eps_perp = np.interp(length_dim, dss_params[:, 0], dss_params[:, 4]) / (length_dim * lp_dna**2)
        eta = np.interp(length_dim, dss_params[:, 0], dss_params[:, 5]) / lp_dna

        return sim_type, eps_bend, eps_par, eps_perp, gamma, eta

    def compute_dE(self, ind0, indf, r_poly_trial, t3_poly_trial, t2_poly_trial,
                   states_trial):
        """
        Compute change in energy of polymer state from proposed new state.
        """
        delta_energy_poly = 0
        # Calculate contribution to polymer energy at the ind0 position
        if ind0 != 0:
            delta_r_trial = r_poly_trial[0, :] - self.r[ind0 - 1, :]
            delta_r_par_trial = np.dot(delta_r_trial, self.t_3[ind0 - 1, :])
            delta_r_perp_trial = delta_r_trial - delta_r_par_trial * self.t_3[ind0 - 1, :]

            delta_r = self.r[ind0, :] - self.r[ind0 - 1, :]
            delta_r_par = np.dot(delta_r, self.t_3[ind0 - 1, :])
            delta_r_perp = delta_r - delta_r_par * self.t_3[ind0 - 1, :]

            bend_vec_trial = (t3_poly_trial[0, :] - self.t_3[ind0 - 1, :]
                            - self.eta * delta_r_perp_trial)
            bend_vec = (self.t_3[ind0, :] - self.t_3[ind0 - 1, :]
                            - self.eta * delta_r_perp)

            delta_energy_poly += (0.5 * self.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                                + 0.5 * self.eps_par * (delta_r_par_trial - self.gamma) ** 2
                                + 0.5 * self.eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
            delta_energy_poly -= (0.5 * self.eps_bend * np.dot(bend_vec, bend_vec)
                                + 0.5 * self.eps_par * (delta_r_par - self.gamma) ** 2
                                + 0.5 * self.eps_perp * np.dot(delta_r_perp, delta_r_perp))

        # Calculate contribution to polymer energy at the indf position
        if indf != self.num_beads:

            delta_r_trial = self.r[indf, :] - r_poly_trial[indf - ind0 - 1, :]
            delta_r_par_trial = np.dot(delta_r_trial, t3_poly_trial[indf - ind0 - 1, :])
            delta_r_perp_trial = delta_r_trial - delta_r_par_trial * t3_poly_trial[indf - ind0 - 1, :]

            delta_r = self.r[indf, :] - self.r[indf - 1, :]
            delta_r_par = np.dot(delta_r, self.t_3[indf - 1, :])
            delta_r_perp = delta_r - delta_r_par * self.t_3[indf - 1, :]

            bend_vec_trial = (self.t_3[indf, :] - t3_poly_trial[indf - ind0 - 1, :]
                            - self.eta * delta_r_perp_trial)
            bend_vec = (self.t_3[indf, :] - self.t_3[indf - 1, :]
                        - self.eta * delta_r_perp)

            delta_energy_poly += (0.5 * self.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                                + 0.5 * self.eps_par * (delta_r_par_trial - self.gamma) ** 2
                                + 0.5 * self.eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
            delta_energy_poly -= (0.5 * self.eps_bend * np.dot(bend_vec, bend_vec)
                                + 0.5 * self.eps_par * (delta_r_par - self.gamma) ** 2
                                + 0.5 * self.eps_perp * np.dot(delta_r_perp, delta_r_perp))

        return delta_energy_poly



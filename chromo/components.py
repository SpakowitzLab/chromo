"""
Components that will make up our simulation.

Various types of polymers, solvents, and other simulation components should be
defined here.
"""
from enum import Enum

import numpy as np
import pandas as pd

from .util import dss_params


class State(Enum):
    """Defines the chemical state of a given bead for a single mark."""

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

    _arrays = 'r', 't3', 't2', 'states', 'bead_length'
    """Track which arrays will be saved to file."""
    _3d_arrays = 'r', 't3', 't2'
    """Track which arrays need to have multi-indexed values (x, y, z)."""

    def __init__(self, name, r, *, bead_length, t3=None, t2=None,
                 states=None, mark_names=None):
        """
        Construct a polymer.

        Parameters
        ----------
        name : str
            A name for convenient repr. Should be a valid filename.
        r : (N, 3) array_like of float
            The positions of each bead.
        t3 : (N, 3) array_like of float
            The tangent vector to each bead in global coordinates.
        t2 : (N, 3) array_like of float
            A material normal to each bead in global coordinates.
        states : (N, M) array_like of int
            State of each of the M epigenetic marks being tracked for each
            bead.
        mark_names : (M, ) str or Sequence[str]
            The name of each chemical modification tracked in `states`, for
            each of tracking which mark is which.
        bead_length : float or (N,) array_like of float
            The amount of polymer path length between this bead and the next
            bead.  For now, a constant value is assumed (the first value if an
            array is passed).
        """
        self.name = name
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.states = states
        self.mark_names = mark_names
        if states is not None:
            # does what atleast_2d(axis=1) should do (but doesn't)
            self.states = self.states.reshape(states.shape[0], -1)
            num_beads, num_marks = self.states.shape
            if num_marks != len(mark_names):
                raise ValueError("Each chemical state must be given a name.")
            if num_beads != len(self.r):
                raise ValueError("Initial epigenetic state of wrong length.")
        # first if makes sure len will work in second if
        if not issubclass(type(bead_length), np.ndarray):
            bead_length = np.atleast_1d(bead_length)
        if len(bead_length) == 1:
            bead_length = np.broadcast_to(bead_length, (self.num_beads, 1))
        self.bead_length = bead_length
        # Load the polymer parameters for the conformational energy
        # for now, the array-values of bead_length are ignored
        self.eps_bend, self.eps_par, self.eps_perp, self.gamma, self.eta \
            = self._find_parameters(self.bead_length[0])

    @classmethod
    def from_csv(cls, csv_file):
        """Construct Polymer from CSV file."""
        df = pd.read_csv(csv_file)
        return cls.from_dataframe(df)

    @classmethod
    def straight_line_in_x(cls, name, num_beads, bead_length, **kwargs):
        """Construct polymer initialized uniformly along the positve x-axis."""
        r = np.zeros((num_beads, 3))
        r[:, 0] = bead_length * np.arange(num_beads)
        t3 = np.zeros((num_beads, 3))
        t3[:, 0] = 1
        t2 = np.zeros((num_beads, 3))
        t2[:, 1] = 1
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    def to_dataframe(self):
        """
        Write canonical CSV representation of the Polymer to file.

        The top-level multiindex values for the columns should correspond to
        the kwargs, to simplify unpacking the structure (and to make for easy
        accessing of the dataframe.
        """
        arrays = {name: self.__dict__[name] for name in self._arrays
                  if self.__dict__[name] is not None}
        vector_arrs = {}
        regular_arrs = {}
        for name, arr in arrays.items():
            if name in self._3d_arrays:
                vector_arrs[name] = arr
            # special-cased below to get correct epigenmark names
            elif name != 'states':
                regular_arrs[name] = arr
        # first construct the parts of the DataFrame that need a multi-index
        # TODO: remove "list" after numpy fixes
        # https://github.com/numpy/numpy/issues/17305
        vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)
        # vector_arr = np.concatenate(vector_arrs.values(), axis=1)
        vector_index = pd.MultiIndex.from_product([vector_arrs.keys(),
                                                   ('x', 'y', 'z')])
        vector_df = pd.DataFrame(vector_arr, columns=vector_index)
        states_index = pd.MultiIndex.from_tuples(
                [('states', name) for name in self.mark_names])
        states_df = pd.DataFrame(self.states, columns=states_index)
        df = pd.concat([vector_df, states_df], axis=1)
        # now throw in the remaining columns one-by-one
        for name, arr in regular_arrs.items():
            df[name] = arr
        return df

    @classmethod
    def from_dataframe(cls, df, name=None):
        """Construct Polymer object from DataFrame. Inverts `.to_dataframe`."""
        # top-level multiindex values correspond to kwargs
        kwnames = np.unique(df.columns.get_level_values(0))
        kwargs = {name: df[name].to_numpy() for name in kwnames}
        # extract names of each epigenetic state from multi-index
        if 'states' in df:
            mark_names = df['states'].columns.to_numpy()
            kwargs['mark_names'] = mark_names
        return cls(name, **kwargs)

    @classmethod
    def from_file(cls, path, name=None):
        """Construct Polymer object from string representation."""
        if name is None:
            name = path.name
        return cls.from_dataframe(
            pd.read_csv(path, header=[0, 1], index_col=0),
            name
        )

    def to_csv(self, path):
        """Save Polymer object to CSV file as DataFrame."""
        return self.to_dataframe().to_csv(path)

    def to_file(self, path):
        """Synonym for *to_csv* to conform to `make_reproducible` spec."""
        return self.to_csv(path)

    @property
    def num_marks(self):
        """Return number of states tracked per bead."""
        return self.states.shape[1]

    @property
    def num_beads(self):
        """Return number of beads in the polymer."""
        return self.r.shape[0]

    def __str__(self):
        """Return string representation of the Polymer."""
        return f"Polymer<{self.name}, nbeads={self.num_beads}, " \
               f"nmarks={self.num_marks}>"

    @staticmethod
    def _find_parameters(length_bead):
        """Look up elastic parameters of ssWLC for each bead_length."""
        lp_dna = 53     # Persistence length of DNA in nm
        # Non-dimensionalized length per bead
        length_dim = length_bead * 0.34 / lp_dna

        # Determine the parameter values using linear interpolation of the
        # parameter table
        eps_bend = np.interp(length_dim, dss_params[:, 0], dss_params[:, 1]) \
            / length_dim
        gamma = np.interp(length_dim, dss_params[:, 0], dss_params[:, 2]) \
            * length_dim * lp_dna
        eps_par = np.interp(length_dim, dss_params[:, 0], dss_params[:, 3]) \
            / (length_dim * lp_dna**2)
        eps_perp = np.interp(length_dim, dss_params[:, 0], dss_params[:, 4]) \
            / (length_dim * lp_dna**2)
        eta = np.interp(length_dim, dss_params[:, 0], dss_params[:, 5]) \
            / lp_dna

        return eps_bend, eps_par, eps_perp, gamma, eta

    def compute_dE(self, ind0, indf, r_poly_trial, t3_poly_trial,
                   t2_poly_trial, states_trial):
        """Compute change in polymer energy moving to proposed new state."""
        delta_energy_poly = 0
        # Calculate contribution to polymer energy at the ind0 position
        if ind0 != 0:
            delta_r_trial = r_poly_trial[0, :] - self.r[ind0 - 1, :]
            delta_r_par_trial = np.dot(delta_r_trial, self.t3[ind0 - 1, :])
            delta_r_perp_trial = delta_r_trial \
                - delta_r_par_trial * self.t3[ind0 - 1, :]

            delta_r = self.r[ind0, :] - self.r[ind0 - 1, :]
            delta_r_par = np.dot(delta_r, self.t3[ind0 - 1, :])
            delta_r_perp = delta_r - delta_r_par * self.t3[ind0 - 1, :]

            bend_vec_trial = (t3_poly_trial[0, :] - self.t3[ind0 - 1, :]
                              - self.eta * delta_r_perp_trial)
            bend_vec = (self.t3[ind0, :] - self.t3[ind0 - 1, :]
                        - self.eta * delta_r_perp)

            delta_energy_poly += (
                0.5 * self.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                + 0.5 * self.eps_par * (delta_r_par_trial - self.gamma) ** 2
                + 0.5 * self.eps_perp * np.dot(delta_r_perp_trial,
                                               delta_r_perp_trial)
            )
            delta_energy_poly -= (
                0.5 * self.eps_bend * np.dot(bend_vec, bend_vec)
                + 0.5 * self.eps_par * (delta_r_par - self.gamma) ** 2
                + 0.5 * self.eps_perp * np.dot(delta_r_perp, delta_r_perp)
            )

        # Calculate contribution to polymer energy at the indf position
        if indf != self.num_beads:

            delta_r_trial = self.r[indf, :] - r_poly_trial[indf - ind0 - 1, :]
            delta_r_par_trial = np.dot(delta_r_trial,
                                       t3_poly_trial[indf - ind0 - 1, :])
            delta_r_perp_trial = delta_r_trial \
                - delta_r_par_trial * t3_poly_trial[indf - ind0 - 1, :]

            delta_r = self.r[indf, :] - self.r[indf - 1, :]
            delta_r_par = np.dot(delta_r, self.t3[indf - 1, :])
            delta_r_perp = delta_r - delta_r_par * self.t3[indf - 1, :]

            bend_vec_trial = (self.t3[indf, :]
                              - t3_poly_trial[indf - ind0 - 1, :]
                              - self.eta * delta_r_perp_trial)
            bend_vec = (self.t3[indf, :] - self.t3[indf - 1, :]
                        - self.eta * delta_r_perp)

            delta_energy_poly += (
                0.5 * self.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                + 0.5 * self.eps_par * (delta_r_par_trial - self.gamma) ** 2
                + 0.5 * self.eps_perp * np.dot(delta_r_perp_trial,
                                               delta_r_perp_trial)
            )
            delta_energy_poly -= (
                0.5 * self.eps_bend * np.dot(bend_vec, bend_vec)
                + 0.5 * self.eps_par * (delta_r_par - self.gamma) ** 2
                + 0.5 * self.eps_perp * np.dot(delta_r_perp, delta_r_perp)
            )

        return delta_energy_poly

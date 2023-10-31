# cython: profile=True

"""Polymers that will make up our simulation.
"""

import pyximport
#from Cython.Includes.numpy import ndarray
#from numpy import ndarray

pyximport.install()

# Built-in Modules
from typing import Callable, Union
from libc.math cimport sin, cos

# External Modules
import numpy as np
cimport numpy as np
import pandas as pd

# Custom Modules
from .util import poly_stat
import chromo.beads as beads
import chromo.util.mc_stat as mc_stat
from chromo.util.linalg cimport (vec_sub3, vec_dot3, vec_scale3)
from .util import dss_params
from .util import poly_paths as paths
from typing import Optional
import chromo.twist_schedule as twist_adjust
# Global type aliases and variables
ctypedef np.uint8_t uint8
cdef np.ndarray empty_2d = np.empty((0, 0))
cdef np.ndarray mty_2d_int = np.empty((0, 0), dtype=int)
cdef np.ndarray empty_1d = np.empty((0, ))
cdef np.ndarray empty_1d_str = np.empty((0, ), dtype=str)


cdef double E_HUGE = 1E99


cdef class TransformedObject:
    """Represents any object undergoing physical transformations during MC sim.

    Notes
    -----
    This class provides the transformation matrix attribute. The transformation
    matrix is updated with each iteration of the MC simulation. It's inclusion
    avoids costly matrix initialization with each iteration, drastically
    improving runtime.

    Attributes
    ----------
    transformation_mat : array_like (4, 4) of double
        Homogeneous transformation matrix for an arbitrary transformation in
        the MC simulation; stored as attribute to avoid costly array
        generation during inner loop of simulation
    """

    def __init__(self):
        self.transformation_mat = np.identity(4, dtype=np.double)


cdef class PolymerBase(TransformedObject):
    """Base class representation of an arbitrary polymer.

    Notes
    -----
    See `TransformedObject` documentation for additional notes and attributes.

    Attributes
    ----------
    name : str
        A name of the polymer for convenient representation; should be a
        valid filename
    log_path : str
        Path to configuration tracker log (if empty, flags for no configuration
        log to be generated)
    beads : beads.Bead
        Monomeric unit forming the polymer
    num_beads : long
        Number of monomeric units forming the polymer
    configuration_tracker : mc_stat.ConfigurationTracker
        Object tracking configuration of the polymer over the course of the
        MC simulation, used for purposes of tracking simulation convergence
    lp : double
        Persistence length of the polymer
    num_binders : long
        Number of reader protein types bound to the polymer which are tracked
        by the simulator
    n_binders_p1 : long
        One more than `num_binders` to avoid repeated calculation
    binder_names : array_like (M,) of str
        Name of M bound chemical states tracked in `states`; length of
        `binder_names` must match number of columns in `states`
    all_inds : array_like (N,) of long
        Indices corresponding to each bead in the polymer
    r, t3, t2 : array_like (N, 3) of double
        Current position (r), tangent vector (t3) and material normal (t2) in
        global coordinates (x, y, z) for each of N beads in the polymer
    r_trial, t3_trial, t2_trial : array_like (N, 3) of double
        Proposed position (r), tangent vector (t3) and material normal (t2) in
        global coordinates (x, y, z) for each of N beads in the polymer;
        evaluated for acceptance using energy change determined by `compute_dE`
    states : array_like (N, M) of long
        Current state of each of the M reader proteins being tracked for each
        of the N beads in the polymer
    states_trial : array_like (N, M) of long
        Trial state of each of the M reader proteins being tracked for each of
        the N beads in the polymer
    chemical_mods : array_like (N, M) of long
        States of M chemical modifications (e.g., counts of H3K9me3) on each
        of N beads in the polymer
    chemical_mod_names : array_like (M,) of str
        Name of M chemical modifications made to the polymer tracked for each
        bead, corresponding to columns in `chemical_mods`
    direction : array_like (3,) of double
        Vector describing the direction of a transformation during a move in
        the MC simulation; stored as attribute to avoid costly array generation
        during inner loop of simulation
    point : array_like (3,) of double
        Used to describe a reference point (typically a fulcrum) for a move
        in the MC simulation; stored as attribute to avoid costly array
        generation during inner loop of simulation
    last_amp_move : double
        Most recent move amplitude during simulation step
    last_amp_bead : long
        Most recent bead amplitude during simulation step
    required_attrs, _arrays, _3d_arrays : array_like (L,) of str
        Minimal required attributes (required_attrs), arrays tracked (_arrays),
        and multi-index arrays tracked (_3d_arrays) by the polymer
    dr, dr_perp, bend : array_like (3,) of double
        Change in position (dr), perpendicular component of the change in
        position (dr_perp), and bend vector (bend) used to compute the
        elastic energy of a bond in a polymer; stored as attribute to avoid
        costly array generation during inner loop of simulation
    dr_test, dr_perp_test, bend_test : array_like (3,) of double
        Change in position (dr_test), perpendicular component of the change in
        position (dr_perp_test), and bend vector (bend_test) of proposed
        configuration used to compute the elastic energy of a bond in a
        polymer; stored as attribute to avoid costly array generation during
        inner loop of simulation
    densities_temp : array_like (2, n_binders_p1, 8) of double
        Densities for current (dim0 = 0) and trial (dim0 = 1) bins containing
        bead during MC move; the densities are tracked for the bead itself
        and each bound reader proteins (stored in dim1); the densities are
        tracked for 8 bins containing the bead before and after the move
        (stored in dim2); stored as attribute to avoid costly array generation
        during inner loop of simulation
    max_binders : long
        Indicates the maximum number of total binders of any type that can bind
        a single bead. This attribute accounts for steric limitations to the
        total number of binders that can attach to any give bead in the polymer.
        A value of `-1` is reserved for no limit to the total number of binders
        bound to a bead. Any move which proposes more than the maximum number
        of binders is assigned an energy cost of 1E99 kT per binder beyond the
        limit.

    Notes
    -----
    Do not instantiate instances of `PolymerBase` -- this class functions as a
    cython equivalent of an abstract base class. `PolymerBase` specifies the
    attributes and methods required to define a new polymer class. The class
    also implements utility methods common to all polymer subclasses.
    """

    def __init__(
        self,
        str name,
        double[:, ::1] r = empty_2d,
        *,
        str log_path = "",
        double[:, ::1] t3 = empty_2d,
        double[:, ::1] t2 = empty_2d,
        double lp = 0,
        long[:, ::1] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        long[:, ::1] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        long max_binders = -1
    ):
        """Any polymer class requires these key attributes.

        Notes
        -----
        Specifying these attributes ensures that new polymer classes are
        compatible with the rest of the codebase. Optional arguments must be
        specified by name.

        `required_attrs` lists the required attributes of any polymer class,
        `_arrays` lists the attributes of the polymer class which are stored as
        arrays, and `_3d_arrays` lists arrays with multi-indexed (x, y, z)
        values. I recognize that these would be better implemented as class
        attributes if we were working in pure Python, but for Cython, it is
        easier to treat these as instance attributes. This is because there is
        no `.__class__` attribute in Cython, and we need to be able to access
        `required_attrs`, `_arrays`, and `_3d_arrays` from the C code for any
        particular subclass using methods in the superclass.

        Parameters
        ----------
        name : str
            A name of the polymer for convenient representation; should be a
            valid filename
        r : array_like (N, 3) of double
            The positions of each bead forming monomeric units of the polymer
        t3 : Optional array_like (N, 3) of double
            The tangent vector to each bead in global coordinates (default is
            empty array)
        t2 : Optional array_like (N, 3) of double
            Material normal to each bead in the global coordinate system; this
            vector should be orthogonal to t2 (default is empty array)
        states : Optional array_like (N, M) of int
            State of each of the M reader proteins being tracked for each of
            the N beads (default is empty array)
        binder_names : Optional array_like (M,) of str
            Name of M bound chemical states tracked in `states`; length of
            `binder_names` must match number of columns in `states` (default is
            empty array)
        chemical_mods : Optional array_like (N, M) of long
            States of M chemical modifications (e.g., counts of H3K9me3) on
            each of N beads in the polymer (default is empty array)
        chemical_mod_names : Optional array_like (M,) of str
            Name of M chemical modifications made to the polymer tracked for
            each bead, corresponding to columns in `chemical_mods` (default
            is empty array)
        log_path : Optional str
            Path to configuration tracker log (default is empty string,
            flagging for no configuration log to be generated)
        max_binders : Optional long
            Indicates the maximum number of total binders of any type that can
            bind a single bead. This attribute accounts for steric limitations
            to the total number of binders that can attach to any give bead in
            the polymer. A value of `-1` is reserved for no limit to the total
            number of binders bound to a bead. Any move which proposes more
            than the maximum number of binders is assigned an energy cost of
            1E99 kT per binder beyond the limit. (default = -1)
        """
        cdef long num_binders, num_beads
        cdef np.ndarray required_attrs, _arrays, _3d_arrays

        super(PolymerBase, self).__init__()

        self.name = name
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.states = states
        self.max_binders = max_binders
        self.binder_names = binder_names
        self.chemical_mods = chemical_mods
        self.chemical_mod_names = chemical_mod_names
        self.num_beads = self.get_num_beads()
        self.fill_missing_arguments()
        self.all_inds = np.arange(0, self.num_beads, 1)
        self.num_binders = self.get_num_binders()
        self.n_binders_p1 = self.num_binders + 1
        self.log_path = log_path
        self.check_binders(states, binder_names)
        self.construct_beads()
        self.update_log_path(log_path)
        self.lp = lp
        self.required_attrs = np.array([
            "name", "r", "t3", "t2", "states", "binder_names", "num_binders",
            "beads", "num_beads", "lp"
        ])
        self._arrays = np.array(['r', 't3', 't2', 'states', 'chemical_mods', 'bead_length'])
        self._3d_arrays = np.array(['r', 't3', 't2'])
        self._single_values = np.array(["name", "lp", "num_beads"])

        # For move proposals
        self.r_trial = self.r.copy()
        self.t3_trial = self.t3.copy()
        self.t2_trial = self.t2.copy()
        self.states_trial = self.states.copy()
        self.transformation_mat = np.empty((4, 4), dtype='d')
        self.last_amp_bead = 0
        self.last_amp_move = 0
        self.direction = np.zeros((3,), dtype='d')
        self.point = np.zeros((3,), dtype='d')

        # For elastic energy calculation
        self.dr = np.zeros((3,), dtype='d')
        self.dr_test = np.zeros((3,), dtype='d')
        self.dr_perp = np.zeros((3,), dtype='d')
        self.dr_perp_test = np.zeros((3,), dtype='d')
        self.bend = np.zeros((3,), dtype='d')
        self.bend_test = np.zeros((3,), dtype='d')

        # For field energy calculation
        self.densities_temp = np.zeros((2, self.n_binders_p1, 8), dtype='d')

    def fill_missing_arguments(self):
        # this definitely works. It just adds default values to things if there is no user input for them.
        """Fill empty arguments in `__init__` call.

        Fills t3, t2, states, binder_names, chemical_mods, chemical_mod_names
        if not defined during polymer initialization.
        """
        if np.size(self.t3) == 0:
            print("No t3 tangent vectors defined.")
            self.t3 = np.zeros((self.num_beads, 3), dtype='d')
        if np.size(self.t2) == 0:
            print("No t2 tangent vectors defined.")
            self.t2 = np.zeros((self.num_beads, 3), dtype='d')
        if np.size(self.states) == 0:
            print("No states defined.")
            self.states = np.zeros((self.num_beads, 1), dtype=int)
        if np.size(self.binder_names) == 0:
            self.binder_names = np.array(["null_reader"])
        if np.size(self.chemical_mods) == 0:
            print("No chemical modifications defined.")
            self.chemical_mods = np.zeros((self.num_beads, 1), dtype=int)
        if np.size(self.chemical_mod_names) == 0:
            self.chemical_mod_names = np.array(["null_mod"])

    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        """Compute the change in configurational (elastic) energy of a polymer.

        Notes
        -----
        This method must be implemented for any subclass of PolymerBase and
        is implemented based on the polymer model representing the subclass.

        Parameters
        ----------
        move_name : str
            Name of the MC move for which the energy change is being calculated
        inds : array_like of int
            Indices of the beads affected by the MC move
        n_inds : int
            Number of indices affected by the MC move

        Returns
        -------
        float
            Elastic energy change associated with the MC move
        """
        #print("does it get to this empty compute_dE function")
        return 0
        #pass

    cdef void construct_beads(self):
        """Construct a list of beads forming the polymer.
        
        Notes
        -----
        This method must be implemented for any subclass of PolymerBase

        Stores a dictionary of beads constructing the polymer chain, where keys
        represent bead IDs and values represent `beads.Bead` objects. The
        dictionary is stored as an attribute of the `PolymerBase` class.
        """
        pass

    def __str__(self):
        """Return string representation of the Polymer.

        Returns
        -------
        str
            String representation listing key attributes of the polymer
        """
        rep = f"Polymer<{self.name}, nbeads={self.num_beads}, " \
            f"nbinder={self.num_binders}>"
        return rep

    def check_attrs(self):
        """Check that the required attributes are specified in the class.

        Notes
        -----
        The required attributes of any polymer are stored in `required_attrs`.
        All required attributes must be initialized for any polymer to be
        instantiated. This method is run during the initialization of the
        polymer to verify that all required attributes are specified.
        """
        cdef str attr
        cdef long i
        for i in range(len(self.required_attrs)):
            attr = str(self.required_attrs[i])
            if not hasattr(self, attr):
                raise NotImplementedError(
                    "Polymer subclass missing required attribute: " + attr
                )
        if self.lp == 0:
            raise ValueError(
                "Specify the persistence length in the subclass of Polymer"
            )

    cdef void check_binders(self, long[:, ::1] states, np.ndarray binder_names):
        """Verify that specified binder states and names are valid.

        Parameters
        ----------
        states : array_like (N, M) of long
            State of each of the M binders being tracked for each of the N bead
        binder_names : array_like (M,) of str
            Name of M bound chemical states tracked in `states`; the number of
            `binder_names` specified must match number of columns in `states`
        """
        cdef long n_beads, num_binders
        if states.shape[1] is not 0:
            num_beads = states.shape[0]
            num_binders = states.shape[1]
            if num_binders != len(binder_names):
                raise ValueError("Each chemical state must be given a name.")
            if num_beads != len(self.r):
                raise ValueError("Initial epigenetic state of wrong length.")

    cdef void check_chemical_mods(
        self,
        long[:, ::1] chemical_mods,
        np.ndarray chemical_mod_names
    ):
        """Check that a valid array of chemical modifications was specified.
        
        Notes
        -----
        There must be a sequence of chemical modifications per reader protein
        included in the model. The sequence of chemical modifications must be
        of length `self.num_beads`, meaning there is a chemical modification
        status specified for each bead of the polymer.

        Parameters
        ----------
        chemical_mods : array_like (N, M) of long
            States of M chemical modifications (e.g., counts of H3K9me3) on
            each of N beads in the polymer
        chemical_mod_names : Optional (M, ) str or Sequence[str]
            Name of chemical modifications made to the polymer tracked for each
            bead in `chemical_mods` attribute
        """
        if self.num_binders != chemical_mods.shape[1]:
            raise ValueError(
                "Number of chemical modifications and binders must match."
            )
        if self.num_beads != chemical_mods.shape[0]:
            raise ValueError(
                "Chemical mod. state must be specified for each polymer bead."
            )
        if len(chemical_mod_names) != chemical_mods.shape[1]:
            raise ValueError(
                "Please specify a name for each chemical modification."
            )
    
    @staticmethod
    def load_seqs(paths_to_seqs: np.ndarray) -> np.ndarray:
        """Load sequence of all chemical modifications on beads of polymer.

        Parameters
        ----------
        paths_to_seqs : array_like (M,) of str
            Paths to the sequence of chemical modifications or binders on the
            beads. A single path should be provided per epigenetic mark.

        Returns
        -------
        array_like (N, M) of long
            States of M chemical modifications (e.g., counts of H3K9me3) on
            each of N beads in the polymer
        """
        num_mods = len(paths_to_seqs)
        if num_mods == 0:
            return mty_2d_int
        first_seq = np.loadtxt(paths_to_seqs[0], dtype=int)
        num_beads = first_seq.shape[0]
        chemical_mods = np.zeros((num_beads, num_mods), dtype=int)
        chemical_mods[:, 0] = first_seq
        for i in range(1, num_mods):
            path = paths_to_seqs[i]
            chemical_mod = np.loadtxt(path, dtype=int)
            if chemical_mod.shape[0] != num_beads:
                raise ValueError(
                    "Inconsistent sequences of chemical modifications"
                )
            chemical_mods[:, i] = chemical_mod
        return chemical_mods
    
    cpdef void update_log_path(self, str log_path):
        """Update the path to the configuration tracker log.
        
        Notes
        -----
        The configuration tracker is used during the simulation to evaluate 
        structural convergence of the polymer.

        Parameters
        ----------
        log_path : str
            Path to the configuration tracker log. If None, no configuration
            tracker will be initialized. (default = None)
        """

        if log_path != "":
            self.configuration_tracker = \
                mc_stat.ConfigurationTracker(log_path, self.r)

    cpdef np.ndarray get_prop(self, long[:] inds, str prop):
        """Get specified property of beads at listed indices.

        Parameters
        ----------
        inds : array_like (L,) of long
            Bead indices at which to isolate multi-indexed attributes (e.g., 
            position, orientation) of the polymer
        prop : str
            Property of the bead to return at specified indices

        Returns
        -------
        np.ndarray (M, 3)
            Specified property of beads at specified indices
        """
        return np.array(
            [self.beads[i].__dict__[prop] for i in inds]
        )

    cpdef np.ndarray get_all(self, str prop):
        """Get some bead property value from all beads.
        
        Notes
        -----
        Generates an array from a specified attribute separately stored in 
        each bead object representing the polymer. This utility function is 
        useful for accessing attributes which are separately stored in bead 
        objects and not aggregated in the polymer object. 

        Parameters
        ----------
        prop : str
            Name of the property to obtain from all beads

        Returns
        -------
        np.ndarray
            Data frame of property values from all beads (default = None)
        """
        array = np.array(
            [self.beads[i].__dict__[prop] for i in range(len(self.beads))]
        )
        if len(array.shape) == 1:
            array = np.atleast_2d(array).T
        return array

    def update_prop(self, str prop):
        """Update bead property for each bead in polymer.

        Notes
        -----
        This utility function copies an attribute of the polymer to each bead
        forming the polymer.

        Parameters
        ----------
        prop : str
            Name of the bead property being updated
        """
        cdef long i
        for i in range(len(self.beads)):
            self.beads[i].__dict__[prop] = self.getattr(self, prop)[i]

    def to_dataframe(self) -> pd.DataFrame:
        """Write canonical CSV representation of the Polymer to file.

        Notes
        -----
        The top-level multi-index values for the columns should correspond to
        the kwargs, to simplify unpacking the structure (and to make for easy
        accessing of the dataframe).

        First get a listing of all the parts of the polymer that have to be
        saved. These are defined in the class's `_arrays` attribute.

        Next, separate out the arrays into two types: "vector" arrays are
        composed of multiple columns (like r) and so we need to build a multi-
        index for putting them into the data frame correctly.

        All other arrays can be added as a single column to the data frame.

        ReaderProtein names is a special case because it does not fit with
        dimensions of the other properties, which are saved per bead.

        Construct the parts of the DataFrame that need a multi-index.

        TODO: remove "list" after numpy fixes issue 17305:
        https://github.com/numpy/numpy/issues/17305

        Replace:
            `vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)`
        with:
            `vector_arr = np.concatenate(vector_arrs.values(), axis=1)`

        After adding multi-index properties, add remaining arrays one-by-one.

        Returns
        -------
        pd.DataFrame
            Data frame representation of the polymer, as recorded in the CSV
            file that was generated
        """
        arrays = {
            name: getattr(self, name) for name in self._arrays
            if hasattr(self, name)
        }
        single_vals = {
            name: getattr(self, name) for name in self._single_values
            if hasattr(self, name)
        }
        vector_arrs = {}
        regular_arrs = {}

        for name, arr in arrays.items():
            if name in self._3d_arrays:
                vector_arrs[name] = arr
            elif name != 'states' and name != 'chemical_mods' and name != "bead_length":
                regular_arrs[name] = arr
        vector_arr = np.concatenate(list(vector_arrs.values()), axis=1)
        vector_index = pd.MultiIndex.from_product(
            [vector_arrs.keys(), ('x', 'y', 'z')]
        )

        vector_df = pd.DataFrame(vector_arr, columns=vector_index)
        vector_df = poly_stat.entry_exit_angle(vector_df)
        #vector_df = poly_stat.extry_exit_angle(vector_df) # added

        if len(self.chemical_mod_names) > 0:
            states_index = pd.MultiIndex.from_tuples(
                [('states', name) for name in self.binder_names]
            )
            chem_mod_index = pd.MultiIndex.from_tuples(
                [('chemical_mods', name) for name in self.chemical_mod_names]
            )
            states_df = pd.DataFrame(
                np.asarray(self.states), columns=states_index
            )
            chemical_mods_df = pd.DataFrame(
                np.asarray(self.chemical_mods), columns=chem_mod_index,
                dtype=int
            )
            bead_length_df = pd.DataFrame(
                np.asarray(self.bead_length),
                dtype=float
            )
            df = pd.concat([vector_df, states_df, chemical_mods_df, bead_length_df], axis=1)
        else:
            df = vector_df
        for name, arr in regular_arrs.items():
            if name == "max_binders":
                df[name] = np.broadcast_to(arr, (len(df.index), 1))
            else:
                #df[name] = np.broadcast_to(arr, (len(df.index), 1))
                df[name] = arr.transpose()
        for name, val in single_vals.items():
            arr_temp = np.empty((len(df.index), 1), dtype=object)
            arr_temp[0, 0] = str(val)

            df[name] = arr_temp
            df[name] = df[name].apply(lambda x: x if x is not None else "")
        return df



    @classmethod
    def from_dataframe(cls, df, name=None):
        """Construct Polymer object from DataFrame; inverts `.to_dataframe`.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame representation of the polymer
        name : Optional str
            Name of the polymer to be formed (default is None)

        Returns
        -------
        PolymerBase
            Polymer object reflecting the dataframe
        """
        # top-level multiindex values correspond to kwargs
        kwnames = np.unique(df.columns.get_level_values(0))
        kwargs = {
            name: np.ascontiguousarray(df[name].to_numpy()) for name in kwnames
        }
        # extract names of each epigenetic state from multi-index
        if 'states' in df:
            binder_names = df['states'].columns.to_numpy()
            kwargs['binder_names'] = binder_names
        if 'chemical_mods' in df:
            chemical_mod_names = df['chemical_mods'].columns.to_numpy()
            kwargs['chemical_mod_names'] = chemical_mod_names
        if 'max_binders' in df:
            kwargs['max_binders'] = kwargs['max_binders'][0]
        if "name" in df and name is None:
            kwargs['name'] = df['name'][0]
        elif name is None:
            kwargs['name'] = "unnamed"
        else:
            kwargs['name'] = name
        if "lp" in df:
            kwargs['lp'] = float(kwargs['lp'][0])
        return cls(**kwargs)

    def to_csv(self, str path) -> pd.DataFrame:
        """Save Polymer object to CSV file as DataFrame.

        Parameters
        ----------
        path : str
            Path to file at which to save CSV representation of the polymer

        Returns
        -------
        pd.DataFrame
            Data frame representation of the polymer, as recorded in the CSV
            file that was generated
        """
        return self.to_dataframe().to_csv(path)

    def to_file(self, str path) -> pd.DataFrame:
        """Synonym for `to_csv` to conform to `make_reproducible` spec.

        Notes
        -----
        See documentation for `to_csv` for details.
        """
        return self.to_csv(path)

    @classmethod
    def from_csv(cls, str csv_file) -> pd.DataFrame:
        """Construct Polymer from CSV file.

        Parameters
        ----------
        csv_file : str
            Path to CSV file from which to construct polymer

        Returns
        -------
        PolymerBase
            Object representation of a polymer
        """
        df = pd.read_csv(csv_file)
        return cls.from_dataframe(df)

    @classmethod
    def from_file(cls, path, name=None):
        """Construct Polymer object from string representation.

        Parameters
        ----------
        path : str
            Path to CSV file representing the polymer
        name : Optional str
            Name of the polymer to be formed (default is None)

        Returns
        -------
        PolymerBase
            Polymer object reflecting the dataframe
        """
        if name is None:
            name = path.split("/")[-1].split(".")[0]
            print("name" + str(name))
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        return cls.from_dataframe(df, name)

    cpdef long get_num_binders(self):
        """Return number of states tracked per bead.
        
        Returns
        -------
        long
            Number of binder types bound to the polymer
        """
        return self.states.shape[1]

    cpdef long get_num_beads(self):
        """Return number of beads in the polymer.
        
        Returns
        -------
        long
            Number of beads forming the polymer
        """
        return self.r.shape[0]

    cpdef bint is_field_active(self):
        """Evaluate if the polymer is affected by a field.
        
        Notes
        -----
        The field will not be active if there are no binders defined or if there
        are no binders bound.

        Returns
        -------
        bint
            Flag indicating whether the polymer is affected by the field (True)
            or is agnostic to the field (False)
        """
        cdef long binders_bound
        if self.states.shape[1] == 0:
            return 0
        binders_bound = np.sum(self.states, dtype=long)
        if binders_bound == 0:
            return 0
        return 1


cdef class Rouse(PolymerBase):
    """Class representation of Rouse polymer.

    Notes
    -----
    The Rouse model represents flexible chains of beads connected by harmonic
    springs. The model inherits from the PolymerBase class, which provides the
    basic attributes and functionality shared by all polymers. Please see
    `PolymerBase` documentation for descriptions of attributes.
    """

    def __init__(
        self,
        str name,
        double[:, ::1] r,
        *,
        double lp,
        np.ndarray[long, ndim=2] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        np.ndarray[long, ndim=2] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = ""
    ):
        """Initialize the Rouse polymer.

        Notes
        -----
        Please refer to documentation for `PolymerBase.__init__()` method for
        description of shared parameters.

        The Rouse model is not yet implemented in this codebase. When the
        model is implemented, be sure to remove the `NotImplementedError`
        statement.

        Parameters
        ----------
        lp : double
            Persistence length of the polymer
        """
        super(Rouse, self).__init__(
            name, r, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names
        )
        self.lp = lp
        raise NotImplementedError(
            "The Rouse polymer model is not yet implemented"
        )

    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        #print("is this function getting used even though it is empty")
        """Compute the change in configurational energy of a polymer.
        
        Notes
        -----
        This method currently serves as a place-holder. The Rouse polymer is 
        not yet implemented into this codebase.
        
        Please see documentation for `PolymerBase.compute_dE()` for 
        description of parameters and return arguments.
        """
        # this is what must be used to calculate energy for SSTWLC which needs to be slightly different than the calculation for SSWLC

        pass

    cdef void construct_beads(self):
        """Construct a list of beads forming the polymer.
        
        Notes
        -----
        This method currently serves as a place-holder. The Rouse polymer is 
        not yet implemented into this codebase.
        """
        pass


cdef class SSWLC(PolymerBase):
    """Class representation of a stretchable, shearable wormlike chain.

    Notes
    -----
    The `SSWLC` class describes a discrete, semiflexible polymer with
    attributes for position, orientation, chemical modification, and reader
    protein binding state.

    The polymer carries around a set of coordinates `r` of shape
    `(num_beads, 3)`, sets of material normals `t3` and `t2`, some number of
    chemical modifications, and some number of chemical binding states per
    bead. A third set of material normals `t1` can be derived from the
    cross-product of the `t2` and `t3`.

    Polymer also carries around a dictionary of beads of length `num_beads`.
    Each bead has a set of coordiantes `r` of length (3,). The beads also
    have material normals `t3` and `t2`. The beads optionally carry around
    some number of chemical modifications and binding states.

    If this codebase is used to simulate DNA, information about the chemical
    properties of each reader protein are stored in `ReaderProtein` objects. To
    model arbitrary SSWLC's, the different types of `Binder` objects may be
    specified to characterize bound components.

    See documentation for `PolymerBase` class for additional paramete details.

    TODO: allow differential discretization, and decay to constant
    discretization naturally.

    Parameters
    ----------
    delta : double
        Spacing between beads of the polymer, non-dimensionalized by
        persistence length
    eps_bend : double
        Bending modulus of polymer (obtained from `_find_parameters`)
    eps_par : double
        Stretch modulus of polymer (obtained from `_find_parameters`)
    eps_perp : double
        Shear modulus of polymer (obtained from `_find_parameters`)
    gamma : double
        Ground-state segment compression (obtained from `_find_parameters`)
    eta : double
        Bend-shear coupling (obtained from `_find_parameters`)
    bead_length : np.ndarray
        Dimensional spacing between beads of the discrete polymer (in nm)
    bead_rad : double
        Dimensional radius of each bead in the polymer (in nm)
    """

    def __init__(
        self, str name, double[:,::1] r, np.ndarray bead_length, *, double lp,
        double bead_rad=5, double[:,::1] t3=empty_2d,
        double[:,::1] t2=empty_2d, long[:,::1] states=mty_2d_int,
        np.ndarray binder_names=empty_1d, long[:,::1] chemical_mods=mty_2d_int,
        np.ndarray chemical_mod_names=empty_1d, str log_path = "",
        long max_binders = -1
    ):
        """Construct a `SSWLC` polymer object as a subclass of `PolymerBase`.

        Notes
        -----
        Please refer to documentation for `PolymerBase.__init__()` method for
        description of shared parameters.

        Parameters
        ----------
        bead_length : ndarray, now it is ndarray
            The amount of polymer path length between subsequent beads (in nm)
        lp : double
            Dimensional persistence length of the SSWLC (in nm)
        bead_rad : double
            Radius of individual beads on the polymer (in nm)
        """
        cdef np.ndarray _arrays
        self.bead_length = bead_length
        super(SSWLC, self).__init__(
            name, r, t3=t3, t2=t2, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names, max_binders=max_binders
        )
        self.bead_rad = bead_rad
        self.construct_beads()
        self.lp = lp
        self._find_parameters(self.bead_length)
        self.required_attrs = np.array([
            "name", "r", "t3", "t2", "states", "binder_names", "num_binders",
            "beads", "num_beads", "lp", "bead_rad"
        ])
        self._arrays = np.array([
            'r', 't3', 't2', 'states', 'bead_length', 'chemical_mods',
            'max_binders', 'bead_length'
        ])
        self.check_attrs()
        self.mu_adjust_factor = 1
        self.temperature_adjust_factor = 1
        self.lt_adjust_factor = 1
        print("mu_adjust")



    cdef void construct_beads(self):
        """Construct `GhostBead` objects forming beads of the polymer.
        """
        self.beads = {
            i: beads.GhostBead(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                bead_length=self.bead_length,
                states=self.states[i],
                binder_names=self.binder_names,
                rad=self.bead_rad
            ) for i in range(len(self.r))
        }

    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    ):
        """Compute change in elastic energy for proposed transformation.
        
        Notes
        -----
        Non-zero elastic energy changes occur at bonds of the polymer 
        directly changed by an MC move. If the indices affected by the MC 
        move are continuous, then the elastic energy change is computed for 
        the bonds at the boundaries of the continuous interval; internal 
        bonds are unchanged by such moves. When the indices affected by an
        MC move are non-continuous (i.e., the move is not applied to a 
        continuous polymer segment), then the elastic energy change is 
        calculated individually at each affected bead.

        Parameters
        ----------
        move_name : str
            Name of the MC move for which the energy change is being calculated
        inds : array_like (N, 3)
            Indices of N beads affected by the MC move
        n_inds : long
            Number of beads affected by the MC move

        Returns
        -------
        delta_energy_poly : double
            Change in polymer elastic energy associated with the trial move
        """
        cdef double delta_energy_poly
        cdef long ind0, indf, ind, i
        #"is this 1037 version of dE the one that gets used")
        delta_energy_poly = 0 #zero change in energy

        if move_name == "change_binding_state":
            if self.is_field_active() == 1:
                ind0 = inds[0]
                indf = inds[n_inds-1] + 1
                #print("anything here at 1044")
                delta_energy_poly += self.binding_dE(ind0, indf, n_inds)
        
        elif (
            move_name == "slide" or move_name == "end_pivot" or
            move_name == "crank_shaft"
        ):
            ind0 = inds[0]
            indf = inds[n_inds-1] + 1
            #print("what about line 1052")
            delta_energy_poly += self.continuous_dE_poly(ind0, indf)
        
        elif move_name == "tangent_rotation": # no changes
            for i in range(n_inds):
                ind = inds[i]
                ind0 = ind
                indf = ind + 1
                #print("is it line 1061")
                delta_energy_poly += self.continuous_dE_poly(ind0, indf)
                #print("does it get this far")
        return delta_energy_poly

    cdef double continuous_dE_poly(
        self,
        long ind0,
        long indf,
    ):
        """Compute change in elastic energy for a continuous bead region.
        
        Notes
        -----
        The internal configuration of a continuous segment selected for a move
        is unaffected; therefore, change in polymer energy can be determined
        from the beads at the ends of the selected segment.

        If a bound of the selected segment exists at the end of the polymer,
        then that bound does not contribute to a change in elastic energy, 
        since there are no bonds continuing beyond an end bead.

        For bounds of affected beads inside the polymer, begin by isolating
        the position and orientation vectors of those bounds and their
        neighbors, then calculate the change in polymer energy for the bead
        pair.

        Paramaters
        ----------
        ind0 : int
            Index of the first bead in the continuous region
        indf : int
            One past the index of the last bead in the continuous region

        Returns
        -------
        delta_energy_poly : double
            Change in energy of polymer associated with trial move
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        ind0_m_1 = ind0 - 1
        indf_m_1 = indf - 1
        # for each pair of beads
        delta_energy_poly = 0
        if ind0 != 0:
            delta_energy_poly += self.bead_pair_dE_poly_forward( # change in energy with segment before change
                self.r[ind0_m_1, :],# evaluates change in energy
                self.r[ind0, :],
                self.r_trial[ind0, :],
                self.t3[ind0_m_1, :],
                self.t3[ind0, :],
                self.t3_trial[ind0, :],
                ind0
            )
        if indf != self.num_beads:
            delta_energy_poly += self.bead_pair_dE_poly_reverse( # change in energy with segment after change
                self.r[indf_m_1, :],
                self.r_trial[indf_m_1, :],
                self.r[indf, :],
                self.t3[indf_m_1, :],
                self.t3_trial[indf_m_1, :],
                self.t3[indf, :],
                #ind0_m_1 # potential bug
                indf
            )

        return delta_energy_poly

    cdef double E_pair(self, double[:] bend, double dr_par, double[:] dr_perp, long ind):
        """Calculate elastic energy for a pair of beads.

        Parameters
        ----------
        bend : double[:]
            Bending vector
        dr_par : double
            Magnitude of the parallel component of the displacement vector
        dr_perp : double[:]
            Perpendicular component of the displacement vector

        Returns
        -------
        double
            Elastic energy of bond between the bead pair
        """
        cdef double E
        E = ( # call on specific values in bead index
            0.5 * self.eps_bend[ind] * vec_dot3(bend, bend) + # change to be arrays calculated for each bond
            0.5 * self.eps_par[ind] * (dr_par - self.gamma[ind]) ** 2 +
            0.5 * self.eps_perp[ind] * vec_dot3(dr_perp, dr_perp)
        )
        return E

    cdef double bead_pair_dE_poly_forward(
        self,
        double[:] r_0, #positon of first bead in move
        double[:] r_1, # position of the second bead in the move
        double[:] test_r_1,
        double[:] t3_0,
        double[:] t3_1,
        double[:] test_t3_1,
        long ind #which bead pair currently working on
    ):


        """Compute change in polymer energy when affecting a single bead pair.
        
        Notes
        -----
        For the current and proposed states of the polymer, calculate change in
        position, as well as the parallel magnitude and perpendicular component
        of that change, of the bead pair. Calculate the bend vectors for the
        existing and trial orientations. Calculate the change in energy given
        these properties using theory for a discretized worm-like chain.
        
        In this 'forward' setup, the bead at index zero is linearly adjacent to
        the affected beads.

        This function can be written in a cleaner (slower) manner, as copied
        below. We have opted for in-place operations where possible to improve
        runtime. This is particularly important because this method is called
        from the inner-most loop of the MC algorithm.

        # Cleaner (and slower) code:
        cdef double de_poly, dr_par, dr_par_test
        cdef long i
        cdef double[:] dr, dr_test, dr_perp, dr_perp_test, bend, bend_test
        dr_test = vec_sub3(test_r_1, r_0)
        dr = vec_sub3(r_1, r_0)
        dr_par_test = vec_dot3(t3_0, dr_test)
        dr_par = vec_dot3(t3_0, dr)
        dr_perp_test = vec_sub3(dr_test, vec_scale3(t3_0, dr_par_test))
        dr_perp = vec_sub3(dr, vec_scale3(t3_0, dr_par))
        bend_test = vec_add3(
            test_t3_1, vec_sub3(
                vec_scale3(t3_0, -1.0), vec_scale3(
                    dr_perp_test, self.eta
                )
            )
        )
        bend = vec_add3(
            t3_1, vec_sub3(
                vec_scale3(t3_0, -1.0), vec_scale3(
                    dr_perp, self.eta
                )
            )
        )
        return self.E_pair(bend_test, dr_par_test, dr_perp_test) -\
            self.E_pair(bend, dr_par, dr_perp)

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        r_1 : array_like (3,)
            Position vector of second bead in bend
        test_r_1 : array_like (3,)
            Position vector of second bead in bend (TRIAL MOVE)
        t3_0 : array_like (3,)
            t3 tangent vector of first bead in bend
        t3_1 : array_like (3,)
            t3 tangent vector of second bead in bend
        test_t3_1 : array_like (3,)
            t3 tangent vector of second bead in bend (TRIAL MOVE)

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef long i
        cdef double dr_par, dr_par_test

        """Adding code segment here to test adding energy from change of entry/exit angles to nucleosome"""
        # Everytime you do a rotation calculation, calculate new entry/exit orientations
        # take XYZ position of nucleosome and translate to position R
        # Rotate by taking T3 and calling that Z, T2 and calling that Y, and TX should be T1


        """End of test code segment"""
        for i in range(3):
            self.dr_test[i] = test_r_1[i] - r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)

        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                test_t3_1[i] - t3_0[i] - self.dr_perp_test[i] * self.eta[ind]
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[ind]
            ) # applying function to find energy in initial and final configuration
        return self.E_pair(self.bend_test, dr_par_test, self.dr_perp_test, ind) -\
            self.E_pair(self.bend, dr_par, self.dr_perp, ind)

    cdef double bead_pair_dE_poly_reverse(
        self,
        double[:] r_0,
        double[:] test_r_0,
        double[:] r_1,
        double[:] t3_0,
        double[:] test_t3_0,
        double[:] t3_1,
        long ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.
        
        Notes
        -----
        For the current and proposed states of the polymer, calculate change in
        position, as well as the parallel magnitude and perpendicular component
        of that change, of the bead pair. Calcualate the bend vectors for the
        existing and trial orientations. Calculate the change in energy given
        these properties using theory for a discretized worm-like chain.

        In this 'reverse' setup, the bead at index one is linearly adjacent to
        the affected beads.

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        test_r_0 : array_like (3,)
            Position vector of first bead in bend (TRIAL MOVE)
        r_1 : array_like (3,)
            Position vector of second bead in bend
        t3_0 : array_like (3,)
            t3 tangent vector of first bead in bend
        test_t3_0 : array_like (3,)
            t3 tangent vector of first bead in bend (TRIAL MOVE)
        t3_1 : array_like (3,)
            t3 tangent vector of second bead in bend

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef double dr_par, dr_par_test
        cdef long i

        for i in range(3):
            self.dr_test[i] = r_1[i]  - test_r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(test_t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)
        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - test_t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                t3_1[i] - test_t3_0[i] - self.dr_perp_test[i] * self.eta[ind]
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[ind]
            )
        return self.E_pair(self.bend_test, dr_par_test, self.dr_perp_test, ind) -\
            self.E_pair(self.bend, dr_par, self.dr_perp, ind)

    cpdef double compute_E(self):
        """Compute the overall polymer energy at the current configuration.
        
        Notes
        -----
        This method loops over each bond in the polymer and calculates the 
        energy change of that bond.

        Returns
        -------
        double
            Configurational energy of the polymer.
        """
        cdef double E, dr_par # energy and magnitude of the parallel component of the displacement vector
        cdef long i, j # initializing the iterators: why is this even necessary
        cdef double[:] dr, dr_perp, bend # magnitude of displacement vector
        cdef double[:] r_0, r_1, t3_0, t3_1 # position of first and second bead in bend, tangent vector of first and second bead in bend
        E = 0 # initialize energy
        for i in range(1, self.num_beads - 1 ): #switching to self.num_beads - 1 from self.num_beads fixed the problem
            r_0 = self.r[i-1, :]
            r_1 = self.r[i, :]
            t3_0 = self.t3[i-1, :]
            t3_1 = self.t3[i, :]
            dr = vec_sub3(r_1, r_0)
            dr_par = vec_dot3(t3_0, dr)
            dr_perp = vec_sub3(dr, vec_scale3(t3_0, dr_par))
            bend = t3_1.copy()
            for j in range(3):
                bend[j] += -t3_0[j] - self.eta[j] * dr_perp[j]
            E += self.E_pair(bend, dr_par, dr_perp, i-1)
        return E

    cdef double binding_dE(self, long ind0, long indf, long n_inds):
        """Compute energy difference associated with change in binding state.

        Parameters
        ----------
        ind0, indf : long
            The first (ind0) and last (indf) indices of the continuous 
            interval for which the binding state was changed
        n_inds : long
            Number of beads affected by binding state move

        Returns
        -------
        double
            Change in energy associated with the proposed binding state move
        """
        cdef long i, ind
        cdef double dE
        
        dE = 0
        for i in range(n_inds):
            ind = ind0 + i
            dE += self.bead_binding_dE(ind, self.states_trial[ind, :])
        return dE

    cdef double bead_binding_dE(self, long ind, long[:] states_trial_ind):
        """Compute energy difference for a single bead change in binding state.
        
        Notes
        -----
        Begin by determining whether or not the bead is chemically modified.
        Then, calculate the energy associated with the proposed binding state.
        For each bound tail, the interaction energy (dependent on the chemical
        modification state) is added to `dE` and the chemical potential is 
        subtracted from `dE`. Finally, using the same approach, subtract the 
        energy associated with the current binding state.
        
        `self.max_binders` indicates the maximum number of total binders of any
        type that can bind a single bead. This attribute accounts for steric
        limitations to the total number of binders that can attach to any give
        bead in the polymer. A value of `-1` is reserved for no limit to the
        total number of binders bound to a bead. Any move which proposes more
        than the maximum number of binders is assigned an energy cost of 1E99
        kT per binder beyond the limit.

        Parameters
        ----------
        ind : long
            Bead index for which the change in binding state is proposed
        states_trial_ind : array_like (M, )
            Proposed binding states for the bead affected by the move
        
        Returns
        -------
        double
            The energy change from the single bead's change in binding state
        """
        cdef long i, num_tails, modified_tails, unmodified_tails
        cdef long tot_bound, diff
        cdef double dE
        cdef long[:] mod_states, states_ind

        mod_states = self.chemical_mods[ind]
        states_ind = self.states[ind].copy()
        dE = 0

        # Evaluate constraint on the total number of binders bound to a bead
        if self.max_binders != -1:

            # Trail configuration
            tot_bound = 0
            for i in range(self.num_binders):
                tot_bound += states_trial_ind[i]
            if tot_bound > self.max_binders:
                diff = tot_bound - self.max_binders
                dE += E_HUGE * diff

            # Current configuration
            tot_bound = 0
            for i in range(self.num_binders):
                tot_bound += states_ind[i]
            if tot_bound > self.max_binders:
                diff = tot_bound - self.max_binders
                dE -= E_HUGE * diff

        # Evaluate the change in binding energy
        for i in range(self.num_binders):
            num_tails = self.beads[ind].binders[i].sites_per_bead
            modified_tails = mod_states[i]
            unmodified_tails = num_tails - modified_tails

            # Trial configuration
            for j in range(states_trial_ind[i]):
                if j+1 <= modified_tails:
                    dE += (
                        self.beads[ind].binders[i].bind_energy_mod -
                        (min(
                            self.beads[ind].binders[i].chemical_potential,
                            self.beads[ind].binders[i].chemical_potential *
                            self.mu_adjust_factor
                        ))
                    )
                else:
                    dE += (
                        self.beads[ind].binders[i].bind_energy_no_mod -
                        (min(
                            self.beads[ind].binders[i].chemical_potential,
                            self.beads[ind].binders[i].chemical_potential *
                            self.mu_adjust_factor
                        ))
                    )

            # Current configuration
            for j in range(states_ind[i]):
                if j+1 <= modified_tails:
                    dE -= (
                        self.beads[ind].binders[i].bind_energy_mod -
                        (min(
                            self.beads[ind].binders[i].chemical_potential,
                            self.beads[ind].binders[i].chemical_potential *
                            self.mu_adjust_factor
                        ))
                    )
                else:
                    dE -= (
                        self.beads[ind].binders[i].bind_energy_no_mod -
                        (min(
                            self.beads[ind].binders[i].chemical_potential,
                            self.beads[ind].binders[i].chemical_potential *
                            self.mu_adjust_factor
                        ))
                    )
        return dE

    def __str__(self):
        """Return string representation of the SSWLC.
        """
        return f"Polymer_Class<SSWLC>, {super(SSWLC, self).__str__()}"

    cpdef void _find_parameters(self, np.ndarray length_bead):
        """Look up elastic parameters of ssWLC for each bead_length.
        
        Interpolate from the parameter table to get the physical parameters of
        the WLC model matching the discretized WLC.
        
        Notes
        -----
        The units of the WLC parameters must be consistent with those in the 
        equation for elastic energy change. Length units are non-dimensionalized
        using the persistence length of the polymer.

        All parameters in the equation for elastic energy change are divided 
        or multiplied by `delta`. To avoid redundant calculations, we divide 
        or multiply the parameters by `delta` here and not each time elastic 
        energy change is calculated.
        
        The definition and units (after division by persistence length) of each 
        physical parameter are listed below:

            - delta         Distance btwn beads (Unitless)
            - eps_bend      Bending modulus (kbT)
            - gamma         Ground-state segment compression (Distance)
            - eps_par       Stretch Modulus (kbT / Distance**2)
            - eps_perp      Shear modulus (kbT / Distance**2)
            - eta           Bend-shear coupling (1 / Distance)

        Parameters
        ----------
        length_bead : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)
        """
        self.delta = length_bead / self.lp # does not need to change b/c array operation
        # initialized eps_bend as array
        self.eps_bend = np.zeros(len(length_bead))
        #self.gamma_bend = np.zeros(len(length_bead))
        self.gamma = np.zeros(len(length_bead))
        self.eps_par = np.zeros(len(length_bead))
        self.eps_perp = np.zeros(len(length_bead))
        self.eta = np.zeros(len(length_bead))
        for i in range (0, len(length_bead)):
            #updating vectors for each bond
            #will only run once, do not need to recalculate every mc
            self.eps_bend[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 1]
            )/ self.delta[i] # interpolate value of physical parameter between two columns
            self.gamma[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 2]
            ) * self.delta[i] * self.lp
            self.eps_par[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 3]
            ) / (self.delta[i] * self.lp**2)
            self.eps_perp[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 4]
            )/ (self.delta[i] * self.lp**2)
            self.eta[i] = np.interp(
                self.delta[i], dss_params[:, 0], dss_params[:, 5]
            ) / self.lp

    @classmethod
    def straight_line_in_x(
        cls, name: str, num_beads: int, bead_length: np.ndarray, **kwargs
    ):
        """Construct polymer initialized uniformly along the positve x-axis.

        Parameters
        ----------
        name : str
            Name of polymer being constructed
        num_beads : int
            Number of monomeric units of polymer
        bead_length : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)

        Returns
        -------
        SSWLC
            Object representation of a polymer currently configured as a
            straight line
        """

        r = np.zeros((num_beads, 3))
        r[:, 1] = np.cumsum(bead_length)
        t3 = np.zeros((num_beads, 3)) # check if this solves anything
        t3[:, 0] = 1
        t2 = np.zeros((num_beads, 3))
        t2[:, 1] = 1
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def arbitrary_path_in_x_y(
        cls, name: str, num_beads: int, bead_length: np.ndarray,
        shape_func: Callable[[float], float], step_size: float = 0.001,
        **kwargs
    ):
        """Construct a polymer initialized as y = f(x) from x = 0.

        Notes
        -----
        As is, this code does not support variable linker lengths.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)
        shape_func : Callable[[float], float]
            Shape of the polymer where z = 0 and y = f(x)
        step_size : float
            Step size for numerical evaluation of contour length when
            building the polymer path

        Returns
        -------
        SSWLC
            Object representing a polymer following path y = f(x)
        """
        r = paths.coordinates_in_x_y(
            num_beads, bead_length, shape_func, step_size
        )
        t3, t2 = paths.get_tangent_vals_x_y(
            r[:, 0], shape_func, step_size
        )
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def arbitrary_path_in_x_y_z(
        cls,
        name: str,
        num_beads: int,
        bead_length: np.ndarray,
        shape_func_x: Callable[[float], float],
        shape_func_y: Callable[[float], float],
        shape_func_z: Callable[[float], float],
        step_size: float = 0.001,
        **kwargs
    ):
        """Construct a polymer initialized as y = f(x) from x = 0.

        Notes
        -----
        As is, this code does not support variable linker lengths.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)
        shape_func_x : Callable[[float], float]
            Parametric functions to obtain the x coordinates of the path
        shape_func_y : Callable[[float], float]
            Parametric functions to obtain the y coordinates of the path
        shape_func_z : Callable[[float], float]
            Parametric functions to obtain the z coordinates of the path
        step_size : float
            Step size for numerical evaluation of contour length when
            building the polymer path

        Returns
        -------
        SSWLC
            Object representing a polymer following path x = X(t), y = Y(t),
            z = Z(t)
        """
        r, parameter_vals = paths.coordinates_in_x_y_z(
            num_beads, bead_length, shape_func_x, shape_func_y,
            shape_func_z, step_size
        )
        t3, t2 = paths.get_tangent_vals_x_y_z(
            parameter_vals, shape_func_x, shape_func_y, shape_func_z,
            step_size, r
        )
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def gaussian_walk_polymer(
        cls, name: str, num_beads: int, bead_length: np.ndarray, **kwargs #, lp: double, **kwargs
    ): # added lp
        """Initialize a polymer to a Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer

        bead_length : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)

        Returns
        -------
        SSWLC
            Object representing a polymer initialized as Gaussian random walk
        """
        r = paths.gaussian_walk(num_beads, bead_length)
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def confined_gaussian_walk(
        cls, name: str, num_beads: int, bead_length: np.ndarray, confine_type: str,
        confine_length: float, **kwargs
    ):
        """Initialize a polymer to a confined Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)
        confine_type : str
            Name of the confining boundary; to indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary. Length
            representation specified in function associated w/ `confine_type`

        Returns
        -------
        Polymer
            Object representing a polymer initialized as a confined Gaussian
            random walk
        """
        r = paths.confined_gaussian_walk(
            num_beads, bead_length, confine_type, confine_length
        )
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    @classmethod
    def confined_uniform_random_walk(
            cls, name: str, num_beads: int, bead_length: np.ndarray, confine_type: str,
            confine_length: float, **kwargs
    ):
        """Initialize a polymer to a confined uniform random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : ndarray
            Dimensional distance between subsequent beads of the polymer (in nm)
        confine_type : str
            Name of the confining boundary; to indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary. Length
            representation specified in function associated w/ `confine_type`

        Returns
        -------
        Polymer
            Object representing a polymer initialized as a confined uniform
            random walk
        """
        r = paths.confined_uniform_random_walk(
            num_beads, bead_length, confine_type, confine_length
        )
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)


cdef class Chromatin(SSWLC):
    """Stretchable, sharable WLC model of chromatin.

    Notes
    -----
    The persistence length of DNA is 53 nm. Each base pair of DNA is 0.34 nm
    along the double helix.

    Please see documentation for `PolymerBase` and `SSWLC` for parameter
    descriptions.

    TODO: This should be instantiated directly from SSWLC. This does not have
    the same level of abstraction as `SSTWLC`, which is also a subclass of
    `SSWLC`. Subclasses at the same level should have the same level of
    abstraction. To implement chromatin as an instance of the SSWLC, we need
    to adjust the `construct_beads` method to accept an arbitrary bead type.
    """

    def __init__(
        self,
        str name,
        np.ndarray[double, ndim=2] r,
        *,
        np.ndarray bead_length,
        double bead_rad = 5,
        np.ndarray[double, ndim=2] t3 = empty_2d,
        np.ndarray[double, ndim=2] t2 = empty_2d,
        np.ndarray[long, ndim=2] states = mty_2d_int,
        np.ndarray binder_names = empty_1d,
        np.ndarray[long, ndim=2] chemical_mods = mty_2d_int,
        np.ndarray chemical_mod_names = empty_1d_str,
        str log_path = "", long max_binders = -1, **kwargs
    ):
        """Construct a `Chromatin` fiber object as a subclass of `SSWLC`.

        Notes
        -----
        The persistence length of DNA is 53 nm. For now, this will be used as
        the persistence length of chromatin, stored in the `lp` attribute.

        Please see documentation for `PolymerBase.__init__()` and
        `SSWLC.__init__()` for parameter descriptions.
        """
        cdef double lp = 53 #why is lp initialized to an actual number, then overwritten?
        super(Chromatin, self).__init__(
            name, r, bead_length=bead_length, bead_rad=bead_rad, lp=lp, t3=t3,
            t2=t2, states=states, binder_names=binder_names, log_path=log_path,
            chemical_mods=chemical_mods, chemical_mod_names=chemical_mod_names,
            max_binders=max_binders
        )
        self.construct_beads()

    cdef void construct_beads(self):
        """Construct Nucleosome objects forming beads of our Chromatin polymer.
        """
        self.beads = {
            i: beads.Nucleosome(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                bead_length=self.bead_length,
                rad=self.bead_rad,
                states=self.states[i],
                binder_names=self.binder_names
            ) for i in range(self.r.shape[0])
        }


cdef class SSTWLC(SSWLC):
    """Class representation of stretchable, shearable wormlike chain w/ twist.

    Notes
    -----
    Please see documentation for `PolymerBase` and `SSWLC` for shared parameter
    descriptions.

    TODO: It does not make sense for Chromatin to be on the same level of
    abstraction as a SSTWLC.

    Parameters
    ----------
    lt : double
        Twist persistence length of the polymer (in nm)
    eps_twist : double
        Twist modulus of the polymer, divided by dimensionless `delta`,
        as appears in the equation for SSTWLC elastic energy (units of kbT)
    """

    def __init__(
            self,
            str name,
            double[:, ::1] r,
            np.ndarray bead_length,
            *,
            double lp,
            double lt,
            double bead_rad = 5,
            double[:, ::1] t3 = empty_2d,
            double[:, ::1] t2 = empty_2d,
            long[:, ::1] states = mty_2d_int,
            np.ndarray binder_names = empty_1d,
            long[:, ::1] chemical_mods = mty_2d_int,
            np.ndarray chemical_mod_names = empty_1d_str,
            str log_path = "", long max_binders = -1,
        ):
        """Construct a `SSTWLC` polymer object as a subclass of `SSWLC`.

        Notes
        -----
        See documentation for `SSWLC` class additional parameter descriptions.

        Parameters
        ----------
        lt : double
            Twist persistence length (in nm)
        """
        cdef np.ndarray _arrays

        self.bead_length = bead_length
        super(SSWLC, self).__init__(
            name, r, t3=t3, t2=t2, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names, max_binders=max_binders
        )
        self.bead_rad = bead_rad
        self.construct_beads()
        self.lp = lp
        self.lt = lt
        self._find_parameters(self.bead_length)
        self.required_attrs = np.array([
            "name", "r", "t3", "t2", "states", "binder_names", "num_binders",
            "beads", "num_beads", "lp", "lt", "bead_rad"
        ])
        self._arrays = np.array(
            ['r', 't3', 't2', 'states', 'bead_length', 'chemical_mods']
        )
        self.check_attrs()


    cpdef void _find_parameters(self, np.ndarray length_bead):
        """Look up elastic parameters of ssWLC for each bead_length.

        Notes
        -----
        See `_find_parameters` method in `SSWLC` class for additional
        documentation.
        
        A new physical parameters, the twist modulus `eps_twist`, is added to
        the SSTWLC model. Once divided by dimensionless `delta`, the twist 
        modulus carries units of kbT.

        TODO: Check definition of `delta`; check units of `eps_twist`.

        Parameters
        ----------
        length_bead : ndarray
            Number of base pairs represented by each bead of the polymer
        """
        self.delta = np.array(length_bead/ self.lp)
        self.eps_bend = np.interp(
            self.delta, dss_params[:, 0], dss_params[:, 1]
        ) / self.delta
        self.gamma = np.interp(
            self.delta, dss_params[:, 0], dss_params[:, 2]
        ) * self.delta * self.lp
        self.eps_par = np.interp(
            self.delta, dss_params[:, 0], dss_params[:, 3]
        ) / (self.delta * self.lp**2)
        self.eps_perp = np.interp(
            self.delta, dss_params[:, 0], dss_params[:, 4]
        ) / (self.delta * self.lp**2)
        self.eta = np.interp(
            self.delta, dss_params[:, 0], dss_params[:, 5]
        ) / self.lp
        self.eps_twist = np.array(self.lt / (self.delta * self.lp))

    cdef double continuous_dE_poly(
        self,
        long ind0,
        long indf,
    ):
        """Compute change in polymer energy for a continuous bead region.

        Notes
        -----
        See documentation for `SSWLC.continuous_dE_poly()` class for details 
        and parameter/return descriptions.
        """
        cdef long ind0_m_1, indf_m_1
        cdef double delta_energy_poly

        ind0_m_1 = ind0 - 1
        indf_m_1 = indf - 1

        delta_energy_poly = 0
        if ind0 != 0:
            delta_energy_poly += self.bead_pair_dE_poly_forward_with_twist(
                self.r[ind0_m_1, :],
                self.r[ind0, :],
                self.r_trial[ind0, :],
                self.t3[ind0_m_1, :],
                self.t3[ind0, :],
                self.t3_trial[ind0, :],
                self.t2[ind0_m_1, :],
                self.t2[ind0, :],
                self.t2_trial[ind0, :],
                ind0
            )
        if indf != self.num_beads:
            delta_energy_poly += self.bead_pair_dE_poly_reverse_with_twist(
                self.r[indf_m_1, :],
                self.r_trial[indf_m_1, :],
                self.r[indf, :],
                self.t3[indf_m_1, :],
                self.t3_trial[indf_m_1, :],
                self.t3[indf, :],
                self.t2[indf_m_1, :],
                self.t2_trial[indf_m_1, :],
                self.t2[indf, :],
                indf_m_1
            )
            #print("does this return energy normally")
        return delta_energy_poly

    cdef double E_pair_with_twist(
        self, double[:] bend, double dr_par, double[:] dr_perp, double omega, long ind
    ):
        """Calculate elastic energy for a pair of beads.
        
        Notes
        -----
        See documentation for `SSWLC.E_pair()` class for more details.

        Parameters
        ----------
        bend : double[:]
            Bending vector
        dr_par : double
            Magnitude of the parallel component of the displacement vector
        dr_perp : double[:]
            Perpendicular component of the displacement vector
        omega : double
            Twist angle (in radians)

        Returns
        -------
        double
            Elastic energy of bond between the bead pair
        """
        delta_omega = omega - (self.bead_length[ind] * 2 * np.pi)/(10.5 * 0.34)
        delta_n = np.round(delta_omega/(2 * np.pi))
        delta_omega = delta_omega - ( 2 * np. pi * delta_n)

        cdef double E
        E = (
            0.5 * self.eps_bend[ind] * vec_dot3(bend, bend) +
            0.5 * self.eps_par[ind] * (dr_par - self.gamma[ind]) ** 2 +
            0.5 * self.eps_perp[ind] * vec_dot3(dr_perp, dr_perp) +
            0.5 * self.eps_twist[ind] * delta_omega**2
        )
        # so we never use the omega value input because of the delta_omega calculation?
        return E

    cdef double bead_pair_dE_poly_forward_with_twist(
        self,
        double[:] r_0,
        double[:] r_1,
        double[:] test_r_1,
        double[:] t3_0,
        double[:] t3_1,
        double[:] test_t3_1,
        double[:] t2_0,
        double[:] t2_1,
        double[:] test_t2_1,
        long ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.

        Notes
        -----
        See documentation for `SSWLC.bead_pair_dE_poly_forward()` class for 
        more details.

        TODO: Check sign of cross product (t2 x t3 OR t3 x t2). Store t1 
        arrays so they don't need to be recreated every time. Manually 
        implement dot products to speed up computation

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        r_1 : array_like (3,)
            Position vector of second bead in bend
        test_r_1 : array_like (3,)
            Position vector of second bead in bend (TRIAL MOVE)
        t3_0, t2_0 : array_like (3,)
            t3, t2 tangent vector of first bead in bend
        t3_1, t2_1 : array_like (3,)
            t3, t2 tangent vector of second bead in bend
        test_t3_1, test_t2_1 : array_like (3,)
            t3, t2 tangent vector of second bead in bend (TRIAL MOVE)

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef long i
        cdef double omega, omega_test, dr_par, dr_par_test
        cdef double[:] t1_0, t1_1, test_t1_1
        #print("does the forward with twist function run")
        t1_0 = np.array([
            t2_0[1] * t3_0[2] - t2_0[2] * t3_0[1],
            t2_0[2] * t3_0[0] - t2_0[0] * t3_0[2],
            t2_0[0] * t3_0[1] - t2_0[1] * t3_0[0]
        ])
        #print("test1")
        t1_1 = np.array([
            t2_1[1] * t3_1[2] - t2_1[2] * t3_1[1],
            t2_1[2] * t3_1[0] - t2_1[0] * t3_1[2],
            t2_1[0] * t3_1[1] - t2_1[1] * t3_1[0]
        ])
        #print("test2")
        test_t1_1 = np.array([
            test_t2_1[1] * test_t3_1[2] - test_t2_1[2] * test_t3_1[1],
            test_t2_1[2] * test_t3_1[0] - test_t2_1[0] * test_t3_1[2],
            test_t2_1[0] * test_t3_1[1] - test_t2_1[1] * test_t3_1[0]
        ])
        #print("this can't be the problem right")
        omega = np.arctan2(
            (np.dot(t2_0, t1_1) - np.dot(t1_0, t2_1)) , # changed / to ,
            (np.dot(t1_0, t1_1) + np.dot(t2_0, t2_1))
        )
        # changing linker length rotates the normal vector, need to have a rotation angle
        #for every bead
        #print("test3")
        omega_test = np.arctan2(
            (np.dot(t2_0, test_t1_1) - np.dot(t1_0, test_t2_1)) , # changed / to ,
            (np.dot(t1_0, test_t1_1) + np.dot(t2_0, test_t2_1))
        )
        #only goes from pi to -pi, not a problem for mc

        for i in range(3):
            #print("test5")
            self.dr_test[i] = test_r_1[i] - r_0[i]
            #print("test6")
            self.dr[i] = r_1[i] - r_0[i]
            #print("test7")
        dr_par_test = vec_dot3(t3_0, self.dr_test)
        #print("test8")
        dr_par = vec_dot3(t3_0, self.dr)
        #print("test9")

        for i in range(3):

            self.dr_perp_test[i] = self.dr_test[i] - t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                test_t3_1[i] - t3_0[i] - self.dr_perp_test[i] * self.eta[ind] # changed to ind from  i
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[ind] # changed to ind from i
            )

        return (self.E_pair_with_twist(
            self.bend_test, dr_par_test, self.dr_perp_test, omega_test, ind
        ) -
            self.E_pair_with_twist(
                self.bend, dr_par, self.dr_perp, omega, ind
        ))

    cdef double bead_pair_dE_poly_reverse_with_twist(
        self,
        double[:] r_0, # Position vector of first bead in bend
        double[:] test_r_0, # Position vector of first bead in bend (TRIAL MOVE)
        double[:] r_1, # Position vector of second bead in bend
        double[:] t3_0, # tangent vector of first bead in bend
        double[:] test_t3_0, #tangent vector of first bead in bend
        double[:] t3_1, #tangent vector of second bead in bend
        double[:] t2_0, # new variable for twist # tangent vector of first bead in bend
        double[:] test_t2_0, # new variable for twist #tangent vector of first bead in bend
        double[:] t2_1, # new variable for twist # tangent vector of second bead in bend
        long ind
    ):
        """Compute change in polymer energy when affecting a single bead pair.

        Notes
        -----
        See documentation for `SSWLC.bead_pair_dE_poly_reverse()` class for 
        more details.

        TODO: Check sign of cross product (t2 x t3 OR t3 x t2). Store t1 
        arrays so they don't need to be recreated every time. Manually 
        implement dot products to speed up computation

        Parameters
        ----------
        r_0 : array_like (3,)
            Position vector of first bead in bend
        test_r_0 : array_like (3,)
            Position vector of first bead in bend (TRIAL MOVE)
        r_1 : array_like (3,)
            Position vector of second bead in bend
        t3_0, t2_0 : array_like (3,)
            t3, t2 tangent vector of first bead in bend
        test_t3_0, test_t2_0 : array_like (3,)
            t3, t2 tangent vector of first bead in bend (TRIAL MOVE)
        t3_1, t2_1 : array_like (3,)
            t3, t2 tangent vector of second bead in bend

        Returns
        -------
        double
            Change in polymer energy for move of a single bead pair
        """
        cdef long i
        cdef double omega, omega_test, dr_par, dr_par_test
        cdef double[:] t1_0, t1_1, test_t1_0
        #print("does reverse with twist function happen")
        t1_0 = np.array([
            t2_0[1] * t3_0[2] - t2_0[2] * t3_0[1],
            t2_0[2] * t3_0[0] - t2_0[0] * t3_0[2],
            t2_0[0] * t3_0[1] - t2_0[1] * t3_0[0]
        ])
        test_t1_0 = np.array([
            test_t2_0[1] * test_t3_0[2] - test_t2_0[2] * test_t3_0[1],
            test_t2_0[2] * test_t3_0[0] - test_t2_0[0] * test_t3_0[2],
            test_t2_0[0] * test_t3_0[1] - test_t2_0[1] * test_t3_0[0]
        ])
        t1_1 = np.array([
            t2_1[1] * t3_1[2] - t2_1[2] * t3_1[1],
            t2_1[2] * t3_1[0] - t2_1[0] * t3_1[2],
            t2_1[0] * t3_1[1] - t2_1[1] * t3_1[0]
        ])
        omega = np.arctan2(
            (np.dot(t2_0, t1_1) - np.dot(t1_0, t2_1)),  # replace / with ,
            (np.dot(t1_0, t1_1) + np.dot(t2_0, t2_1))
        )
        omega_test = np.arctan2(
            (np.dot(test_t2_0, t1_1) - np.dot(test_t1_0, t2_1)) , # replace / with ,
            (np.dot(test_t1_0, t1_1) + np.dot(test_t2_0, t2_1))
        )
        for i in range(3):
            self.dr_test[i] = r_1[i]  - test_r_0[i]
            self.dr[i] = r_1[i] - r_0[i]
        dr_par_test = vec_dot3(test_t3_0, self.dr_test)
        dr_par = vec_dot3(t3_0, self.dr)
        for i in range(3):
            self.dr_perp_test[i] = self.dr_test[i] - test_t3_0[i] * dr_par_test
            self.dr_perp[i] = self.dr[i] - t3_0[i] * dr_par
            self.bend_test[i] = (
                t3_1[i] - test_t3_0[i] - self.dr_perp_test[i] * self.eta[ind] # changed i to ind
            )
            self.bend[i] = (
                t3_1[i] - t3_0[i] - self.dr_perp[i] * self.eta[ind] #changed i to ind
            )
        return (self.E_pair_with_twist(
            self.bend_test, dr_par_test, self.dr_perp_test, omega_test, ind
        ) -
            self.E_pair_with_twist(
                self.bend, dr_par, self.dr_perp, omega, ind
        ))


cdef class LoopedSSTWLC(SSTWLC):
    """Class representation of a looped SSTWLC.
    """

    def __init__(
        self,
        str name, double[:,::1] r, *, np.ndarray bead_length, double lp=53,
        double lt=46.37, double bead_rad=5, double[:,::1] t3=empty_2d,
        double[:,::1] t2=empty_2d, long[:,::1] states=mty_2d_int,
        np.ndarray binder_names=empty_1d, long[:,::1] chemical_mods=mty_2d_int,
        np.ndarray chemical_mod_names=empty_1d, str log_path = "",
        long max_binders = -1
    ):
        """Construct a looped SSTWLC object as a subclass of `SSTWLC`.

        Notes
        -----
        See documentation for `SSTWLC` class for description of parameters.

        TODO: Change the default values of lt, lp to ones that makes more sense
        """
        super(LoopedSSTWLC, self).__init__(
            name, r, bead_length=bead_length, lp=lp, lt=lt, bead_rad=bead_rad,
            t3=t3, t2=t2, states=states, binder_names=binder_names,
            log_path=log_path, chemical_mods=chemical_mods,
            chemical_mod_names=chemical_mod_names, max_binders=max_binders
        )

    @classmethod
    def looped_confined_gaussian_walk(
        cls,
        name: str,
        num_beads: int,
        bead_length: np.ndarray,
        confine_type: str,
        confine_length: float,
        **kwargs
    ):
        """Initialize a polymer to a looped, confined Gaussian random walk.

        Parameters
        ----------
        name : str
            Name of the polymer
        num_beads : int
            Number of monomer units on the polymer
        bead_length : ndarray
            The amount of polymer path length between this bead and the next
            bead. For now, a constant value is assumed (the first value if an
            array is passed).
        confine_type : str
            Name of the confining boundary. To indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary. Length
            representation specified in function associated w/ `confine_type`

        Returns
        -------
        SSTWLC
            Object representing a SSTWLC initialized as a confined, looped
            Gaussian random walk
        """
        r = paths.looped_confined_gaussian_walk(
            num_beads, bead_length, confine_type, confine_length
        )
        t3, t2 = paths.estimate_tangents_from_coordinates(r)
        return cls(name, r, t3=t3, t2=t2, bead_length=bead_length, **kwargs)

    cdef void construct_beads(self):
        """Construct Nucleosome objects forming beads of our polymer.
        """
        self.beads = {
            i: beads.Nucleosome(
                id_=i,
                r=self.r[i],
                t3=self.t3[i],
                t2=self.t2[i],
                bead_length=self.bead_length,
                rad=self.bead_rad,
                states=self.states[i],
                binder_names=self.binder_names
            ) for i in range(self.r.shape[0])
        }

    cdef double continuous_dE_poly(
        self,
        long ind0, #
        long indf,
    ):
        """Compute change in polymer energy for a continuous bead region.

        Notes
        -----
        See documentation for this method in `SSWLC` class for details and 
        parameter/return descriptions.
        """
        cdef long ind0_m_1, indf_m_1 # ind0 -1 , one after things affected by move
        cdef double delta_energy_poly

        # Elastic energy change from LHS of the move
        delta_energy_poly = 0
        if ind0 != 0:
            ind0_m_1 = ind0 - 1
        else:
            ind0_m_1 = self.num_beads - 1
        delta_energy_poly += self.bead_pair_dE_poly_forward(
            self.r[ind0_m_1, :],
            self.r[ind0, :],
            self.r_trial[ind0, :],
            self.t3[ind0_m_1, :],
            self.t3[ind0, :],
            self.t3_trial[ind0, :],
            ind0
        )
        # Elastic energy change from RHS of the move
        indf_m_1 = indf - 1
        if indf == self.num_beads:
            indf = 0
        delta_energy_poly += self.bead_pair_dE_poly_reverse(
            self.r[indf_m_1, :],
            self.r_trial[indf_m_1, :],
            self.r[indf, :],
            self.t3[indf_m_1, :],
            self.t3_trial[indf_m_1, :],
            self.t3[indf, :],
            #ind0_m_1 # suspected bug location
            indf_m_1
        )

        return delta_energy_poly


cpdef double sin_func(double x):
    """Sine function to which the polymer will be initialized.

    Parameters
    ----------
    x : double
        Input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    return 50 * sin(x / 35)


cpdef double helix_parametric_x(double t):
    """Parametric equation for x-coordinates of a helix.

    Parameters
    ----------
    t : double
        Parameter input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    cdef double x = 60 * cos(t)
    return x


cpdef double helix_parametric_y(double t):
    """Parametric equation for y-coordinates of a helix.

    Parameters
    ----------
    t : double
        Parameter input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    cdef double y = 60 * sin(t)
    return y


cpdef double helix_parametric_z(double t):
    """Parametric equation for z-coordinates of a helix.

    Parameters
    ----------
    t : double
        Parameter input to the shape function

    Returns
    -------
    double
        Output to the shape function
    """
    cdef double z = 20 * t
    return z

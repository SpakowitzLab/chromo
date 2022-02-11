"""Fields discretize space to efficiently track changes in Mark energy.

Notes
-----
Creates a field object that contains parameters for the field calculations
and functions to generate the densities.

Cythonized to accomodate expensive `compute_dE` method.

To profile runtime of cython functions, add the following comment to the top of
this file: # cython: profile=True
"""

import pyximport
pyximport.install()

# Built-in Modules
from pathlib import Path
import sys
from libc.math cimport floor, sqrt

# External Modules
import numpy as np
cimport numpy as np
import pandas as pd

# Custom Modules
import chromo.polymers as poly
cimport chromo.polymers as poly
from chromo.util.linalg cimport vec_dot3


cdef double E_HUGE = 1E99

cdef list _field_descriptors = [
    'x_width', 'nx', 'y_width', 'ny', 'z_width', 'nz', 'confine_type',
    'confine_length'
]
cdef list _int_field_descriptors = ['nx', 'ny', 'nz']
cdef list _str_field_descriptors = ['confine_type']
cdef list _float_field_descriptors = [
    'x_width', 'y_width', 'z_width', 'confine_length'
]


cdef class FieldBase:
    """A discretization of space for computing energies.

    Notes
    -----
    Must be subclassed to be useful.

    As implemented so far, this codebase accepts only one polymer in the field

    Attributes
    ----------
    polymers : List[PolymerBase]
        Polymers contained in the field; for now, this codebase accepts only
        one polymer in the field
    n_polymers : long
        Number of polymers in the field
    marks : pd.DataFrame
        Table containing marks bound to the polymer and their relevant
        properties
    """

    @property
    def name(self):
        """Print the name of the field.

        Notes
        -----
        For now, there's only one field per sim, so classname works.
        """
        return self.__class__.__name__

    def __init__(self):
        """Construct a field holding no Polymers, tracking no Marks.
        """
        self.polymers = []
        self.n_polymers = 0
        self.marks = pd.DataFrame()

    def __str__(self):
        """Print representation of empty field.
        """
        return "Field<>"

    def __contains__(self, poly):
        """Check if a polymer is currently set to interact with this field.

        Notes
        -----
        No polymers interact with the base field object.
        """
        return self.polymers.__contains__(poly)

    def compute_dE(self, poly, inds, n_inds, packet_size) -> float:
        """Compute the change in field energy due to a proposed move.

        Parameters
        ----------
        poly : `chromo.PolymerBase`
            The polymer which has been moved
        inds : array_like (N,)
            Indices of beads being moved
        n_inds : long
            Number of beads affected by the move
        packet_size : long
            Number of points to average together when calculating the field
            energy change of a move; done to reduce the computational expense
            of the field energy calculation (at the expense of precision)

        Returns
        -------
        double
            The change in energy caused by the Polymer's movement in this
            field.
        """
        pass


class Reconstructor:
    """Defer construction of `Field` until after `PolymerBase`/`Mark` instances.

    Notes
    -----
    Constructs a kwargs object that can be re-passed to the appropriate `Field`
    constructor when they become available.

    Parameters
    ----------
    field_constructor : cls
        Class from which the field will be instantiated
    kwargs : Dict
        Keyword arguments used to instantiate the field
    """

    def __init__(self, cls, **kwargs):
        """Construct our Reconstructor.

        Parameters
        ----------
        field_constructor : cls
            Class from which the field will be instantiated
        """
        self.field_constructor = cls
        self.kwargs = kwargs

    def finalize(self, polymers, marks) -> FieldBase:
        """Finish construction of appropriate `Field` object.

        Parameters
        ----------
        polymers : List[PolymerBase]
            List of polymers contained in the field
        marks : pd.DataFrame
            Table representing chemical marks bound to polymers in the field

        Returns
        -------
        FieldBase
            Field representation of discretized space containing polymers
        """
        return self.field_constructor(
            polymers=polymers, marks=marks, **self.kwargs
        )

    @classmethod
    def from_file(cls, path: Path) -> FieldBase:
        """Assume class name is encoded in file name.

        Parameters
        ----------
        path : Path
            File path object directed to the file defining the field

        Returns
        -------
        FieldBase
            Field representation of discretized space containing polymers
        """
        constructor = globals()[path.name]
        kwargs = pd.read_csv(path).iloc[0].to_dict()
        return cls(constructor, **kwargs)

    def __call__(self, polymers, marks) -> FieldBase:
        """Synonym for `Reconstructor.finalize()`.

        Notes
        -----
        See documentation for `Reconstructor.finalize()` for additional
        details and parameter/returns definitions.
        """
        return self.finalize(polymers, marks)


cdef class UniformDensityField(FieldBase):
    """Rectilinear discretization of space as a rectangular box.

    Notes
    -----
    Each bead contributes "mass" to the eight nearest voxels containing it.
    The mass is linearly interpolated based on the distance of the bead from
    the center of each voxel. This is much more stable numerically than just
    using the voxels as "bins," as would be done in e.g., finite-difference
    discretization.

    Many attributes are updated during each calculation of field energy
    change. We choose to track and update these attributes to avoid costly
    array initialization during each iteration of the MC simulation.
    Affected attributes are identified with the "TEMP" flag in their
    description.

    Attributes
    ----------
    _field_descriptors : List[str]
        Names of attributes which describe the geometric layout of the field
        as a grid of voxels
    x_width, y_width, z_width : double
        Dimensions of the discretized field in the x, y, and z directions
    width_xyz : double[:]
        Vectorized representation of the x, y, and z widths of the
        discretized field
    nx, ny, nz : long
        Number of voxels in each of the x, y, and z directions
    dx, dy, dz : double
        Width of each voxel in the x, y, and z directions
    dxyz : double[:]
        Vectorized representation of the voxel widths in the x, y, and z
        directions
    n_bins : long
        Number of voxels in the field
    vol_bin : double
        Volume of each voxel; equal to `dx*dy*dz`
    bin_index : long[:, ::1]
        For each of the `nx*ny*nz` voxels in the field, stores indices
        identifying eight vertices at the corners of the voxel
    nbr_indices_with_trial : long[:, ::1]
        Super-indices of eight neighboring voxels containing a bead's
        position in space (dim1) from the polymer's current configuration
        (dim1=0) and proposed configuration (dim1=1); TEMP
    nbr_inds : long[:]
        Super-indices of eight neighboring voxels containing a bead's
        position in space; TEMP
    index_xyz_with_trial : long[:, ::1]
        Vector containing a bead's bin index in the x, y, and z directions
        (dim1) for the polymer's current configuration (dim0=0) and trial
        configuration (dim0=1); TEMP
    index_xyz : long[:]
        Vector containing a bead's bin index in the x, y, and z directions; TEMP
    wt_vec_with_trial : double [:]
        Bead's weight linearly interpolated between the eight nearest voxels
        surrounding it (dim1) for a polymer's current configuration (dim0=0)
        and trial configuration (dim0=1); TEMP
    wt_vec : double[:]
        Vector containing a bead's weight linearly interpolated between the
        eight nearest voxels surrounding it; TEMP
    xyz_with_trial : double[:, ::1]
        Position of the bead in the x, y, z directions (dim1), shifted during
        linear interpolation of voxel weights for the polymer's current
        configuration (dim0=0) and trial configuration (dim0=1); TEMP
    xyz : double[:]
        Position of the bead in the x, y, z directions, shifted during linear
        interpolation of voxel weights; TEMP
    weight_xyz_with_trial : double[:, ::1]
        Vector of a bead's linearly interpolated weights in a voxel in the x, y,
        and z directions as determined for the voxel surrounding the bead with
        the lowest x, y, z indices (dim1) for the polymer's current
        configuration (dim0=0) and tiral configuration (dim0=1); TEMP
    weight_xyz : double[:]
        Vector of a bead's linearly interpolated weights in a voxel in the x, y,
        and z directions as determined for the voxel surrounding the bead with
        the lowest x, y, z indices; TEMP
    num_marks : long
        Number of chemical marks bound to the polymer in the simulation
    doubly_bound, doubly_bound_trial : long[:]
        Vectors indicating whether or not a bead is doubly bound by each
        tracked chemical mark in the current polymer configuration
        (`doubly_bond`) and trial configuration (`doubly_bound_trial`); TEMP
    confine_type : str
        Name of the confining boundary around the polymer; if the polymer is
        unconfined, `confine_type` is a blank string
    confine_length : double
        Length scale of the confining boundary
    density, density_trial : double[:, ::1]
        Current (`density`) and proposed (`density_trials`) density of beads
        (column 0) and each chemical mark (columns 1...) in each voxel of the
        discretized space; voxels are sorted by super-index and arranged down
        the rows the the density arrays
    total_mark_densities : dict[str, double]
        Total density of each mark in all voxels affected by an MC move.
    access_vols : Dict[long, double]
        Mapping of voxel super-index (keys) to volume of the voxel inside the
        confining boundary (values).
    chi : double
        Negative local Flory-Huggins parameter dictating non-specific bead
        interaction
    dict_ : dict
        Dictionary of key attributes defining the field and their values
    mark_dict : dict
        Dictionary representation of each chemical mark and their properties
    half_width_xyz : double[:]
        Half the width of the full discretized space in the x, y,
        and z directions; equivalent to `np.array([self.x_width/2,
        self.y_width/2, self.z_width/2])`
    half_step_xyz : double[:]
        Half the length of the voxel in single x, y, and z directions;
        equivalent to `np.array([self.dx/2, self.dy/2, self.dz/2])`
    n_xyz_m1 : long[:]
        One less than the number of voxels that fit in the x, y, and z
        directions of the  field.
    inds_xyz_to_super : long[:, :, ::1]
        Array containing the super-index of each voxel (stored in the array
        values) for each combination of x-voxel position (dim0), y-voxel
        position (dim1) and z-voxel position (dim2).
    """

    def __init__(
        self, polymers, marks, x_width, nx, y_width, ny, z_width, nz,
        confine_type = "", confine_length = 0.0, chi = 1.0
    ):
        """Construct a `UniformDensityField` containing polymers.

        Parameters
        ----------
        polymers : List[PolymerBase]
            List of polymers contained in the field
        marks : pd.DataFrame
            Output of `chromo.marks.make_mark_collection` applied to the list
            of `Mark` objects contained in the field
        x_width, y_width, z_width : double
            Width of the box containing the field in the x, y, and z-directions
        nx, ny, nz : long
            Number of bins in the x, y, and z-directions
        confine_type : str
            Name of the confining boundary; to indicate model w/o confinement,
            enter a blank string for this argument
        confine_length : double
            The lengthscale associated with the confining boundary; length
            representation specified in function associated w/ `confine_type`
        chi : double
            Negative local Flory-Huggins parameter dictating non-specific bead
            interaction (default = 1)
        """
        self._field_descriptors = _field_descriptors
        self.polymers = polymers
        self.n_polymers = len(polymers)
        self.marks = marks
        for poly in polymers:
            if poly.num_marks != len(marks):
                raise NotImplementedError(
                    "For now, all polymers must use all of the same marks."
                )
        self.x_width = x_width
        self.y_width = y_width
        self.z_width = z_width
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.init_grid()
        self.num_marks = len(marks)
        self.doubly_bound = np.zeros((self.num_marks,), dtype=int)
        self.doubly_bound_trial = np.zeros((self.num_marks,), dtype=int)
        self.init_field_energy_prefactors()
        self.density = np.zeros((self.n_bins, self.num_marks + 1), 'd')
        self.density_trial = self.density.copy()
        self.confine_type = confine_type
        self.confine_length = confine_length
        self.access_vols = self.get_accessible_volumes(n_side=20)
        self.chi = chi
        self.dict_ = self.get_dict()
        self.mark_dict = self.marks.to_dict(orient='records')
        self.update_all_densities_for_all_polymers()
    
    def init_grid(self):
        """Initialize the discrete grid containing the field.

        Notes
        -----
        The field will be represented by a grid of rectangular prism (typically
        cubical) voxels, each of dimensions `dx` by `dy` by `dz`. To calculate
        interaction energies, density will be determined inside each voxel and
        a mean field approach will be applied.
        """
        self.dx = self.x_width / self.nx
        self.dy = self.y_width / self.ny
        self.dz = self.z_width / self.nz
        self.dxyz = np.array([self.dx, self.dy, self.dz])
        self.n_bins = self.nx * self.ny * self.nz
        self.vol_bin = self.x_width * self.y_width * self.z_width / self.n_bins
        self.bin_index = UniformDensityField._get_corner_bin_index(
            self.nx, self.ny, self.nz
        )
        self.width_xyz = np.array([self.x_width, self.y_width, self.z_width])
        self.half_width_xyz = np.array(
            [0.5 * self.x_width, 0.5 * self.y_width, 0.5 * self.z_width]
        )
        self.half_step_xyz = np.array(
            [0.5 * self.dx, 0.5 * self.dy, 0.5 * self.dz]
        )
        self.n_xyz_m1 = np.array(
            [self.nx - 1, self.ny - 1, self.nz - 1]
        )
        self.precompute_ind_xyz_to_super()
        self.nbr_inds = np.empty((8,), dtype=int)
        self.nbr_inds_with_trial = np.empty((2, 8), dtype=int)
        self.wt_vec = np.empty((8,), dtype='d')
        self.wt_vec_with_trial = np.empty((2, 8), dtype='d')
        self.xyz = np.empty((3,), dtype='d')
        self.xyz_with_trial = np.empty((2, 3), dtype='d')
        self.weight_xyz = np.empty((3,), dtype='d')
        self.weight_xyz_with_trial = np.empty((2, 3), dtype='d')
        self.index_xyz = np.empty((3,), dtype=int)
        self.index_xyz_with_trial = np.empty((2, 3), dtype=int)

    cdef void precompute_ind_xyz_to_super(self):
        """Precompute how voxel (x, y, z) indices translate to superindices.
        """
        cdef long i, j, k

        self.inds_xyz_to_super = np.empty(
            (self.nx+1, self.ny+1, self.nz+1), dtype=int
        )
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                for k in range(self.nz + 1):
                    self.inds_xyz_to_super[i, j, k] = inds_to_super_ind(
                    i % self.nx, j % self.ny, k % self.nz, self.nx, self.ny
                )
    
    def init_field_energy_prefactors(self):
        """Initialize the field energy prefactor for each epigenetic mark.
        """
        mark_names = []
        for i in range(self.num_marks):
            mark_name = self.marks.loc[i, "name"]
            mark_names.append(mark_name)

        for i in range(self.num_marks):
            self.marks.at[i, 'field_energy_prefactor'] = (
                0.5 * self.marks.iloc[i].interaction_energy
                * self.marks.iloc[i].interaction_volume
                * self.vol_bin
            )

            self.marks.at[i, 'interaction_energy_intranucleosome'] = (
                self.marks.iloc[i].interaction_energy
                * (1 - self.marks.iloc[i].interaction_volume / self.vol_bin)
            )

            for next_mark in mark_names:
                if next_mark in self.marks.iloc[i].cross_talk_interaction_energy.keys():
                    self.marks.at[i, 'cross_talk_field_energy_prefactor'][next_mark] = (
                        self.marks.iloc[i].cross_talk_interaction_energy[next_mark]
                        * self.marks.iloc[i].interaction_volume
                        * self.vol_bin
                    )
                else:
                    self.marks.at[i, 'cross_talk_field_energy_prefactor'][next_mark] = 0


    cpdef dict get_accessible_volumes(self, long n_side):
        """Numerically find accessible volume of voxels at confinement edge.

        Parameters
        ----------
        n_side : long
            When a voxel is determined to be partially contained by the
            confinement, we will numerically determine the accessible volume in
            that voxel. To do this, we will place a cubic grid of points within
            the voxel, and we will count the number of grid points within the
            confinement. The parameter `n_side` gives the number of grid points
            to include on each side of the cubic grid. For example, if n_side =
            10, then we will place a 10x10x10 grid of points in every partially
            contained voxel to determine the accessible volume of that voxel.

        Returns
        -------
        access_vol : dict
            Mapping of bin super index to accessible volume inside the 
            confining boundary
        """
        cdef long i, voxel
        cdef long[:] split_voxels
        cdef long[:, ::1] xyz_inds
        cdef double[:, ::1] xyz_coords, dxyz_point
        cdef dict access_vols
        
        access_vols = {i : self.vol_bin for i in range(self.n_bins)}
        if self.confine_type != "":
            xyz_inds = np.zeros((self.n_bins, 3), dtype=int)
            for i in range(self.n_bins):
                xyz_inds[i, :] = super_ind_to_inds(i, self.nx, self.ny)
            xyz_coords = self.get_voxel_coords(xyz_inds)
            buffer_dist = np.sqrt(2) / 4 * max(self.dx, self.dy, self.dz)
            split_voxels = self.get_split_voxels(xyz_coords, buffer_dist)
            dxyz_point = self.define_voxel_subgrid(n_side)
            for i in range(self.n_bins):
                if split_voxels[i] == 1:
                    access_vols[i] = \
                        self.get_frac_accessible(xyz_coords[i], dxyz_point) *\
                        self.vol_bin
        return access_vols
            
    cdef double[:, ::1] get_voxel_coords(self, long[:, ::1] xyz_inds):
        """Get voxel coordinates from xyz bin indices.

        Parameters
        ----------
        xyz_inds : long[:, ::1]
            Indices in the x, y, and z directions for each bin, where rows
            represent individual bins and columns indicate the index position
            of the bin in the x, y, and z directions.

        Returns
        -------
        double[:, ::1]
            Coordinates in the x, y, and z directions for each bin, where rows
            represent individual bins and columns indicate coordinates in the
            x, y, and z directions.
        """
        cdef long i, j
        cdef long[:] nxyz
        cdef double[:, ::1] dxyz, centralized_inds, xyz_coords
        
        nxyz = np.array([self.nx, self.ny, self.nz])
        dxyz = np.array([
                [self.dx, 0, 0],
                [0, self.dy, 0],
                [0, 0, self.dz]
        ])
        centralized_inds = np.zeros((self.n_bins, 3))
        for i in range(self.n_bins):
            for j in range(3):
                centralized_inds[i, j] = xyz_inds[i, j] - (nxyz[j] - 1) / 2
        xyz_coords = np.matmul(centralized_inds, dxyz)
        return xyz_coords
    
    cdef long[:] get_split_voxels(
        self, double[:, ::1] xyz_coords, double buffer_dist
    ):
        """Identify voxels intersecting with the confinement boundary.

        Parameters
        ----------
        xyz_coords : double[:, ::1]
            Cartesian coordinates identifying the center of each voxel, where
            rows represent individual voxels and columns indicate x, y, and z
            positions at the center of the voxel
        buffer_dist : double
            Distance from confining boundary at which a voxel is marked as 
            "split" and is assessed as potentially intersecting the boundary

        Returns
        -------
        long[:]
            Array of super-indices cooresponding to voxels that intersect with
            the confinement
        """
        cdef long[:] split_voxels = np.ones(self.n_bins, dtype=int)
        
        for i in range(self.n_bins):
            dist = sqrt(
                xyz_coords[i, 0] ** 2 +
                xyz_coords[i, 1] ** 2 +
                xyz_coords[i, 2] ** 2
            )
            if (dist < self.confine_length - buffer_dist or 
                dist > self.confine_length + buffer_dist):
                split_voxels[i] = 0
        return split_voxels
    
    cdef double[:, ::1] define_voxel_subgrid(self, long n_pt_side):
        """Define voxel subgrid for accessible volume calculation.

        Parameters
        ----------
        n_pt_side : long
            When a voxel is determined to be partially contained by the
            confinement, we will numerically determine the accessible volume in
            that voxel. To do this, we will place a cubic grid of points within
            the voxel, and we will count the number of grid points within the
            confinement. The parameter `n_side` gives the number of grid points
            to include on each side of the cubic grid. For example, if n_side =
            10, then we will place a 10x10x10 grid of points in every partially
            contained voxel to determine the accessible volume of that voxel.

        Returns
        -------
        double[:, ::1]
            Array of subgrid points (relative to origin) for determination of
            voxel accessible volume. Rows indicate individual points, and 
            columns indicate x, y, z coordinates for each point.
        """
        cdef long total_points
        cdef double dx_point, dy_point, dz_point
        cdef double[:, ::1] dxyz_point

        total_points = n_pt_side ** 3
        dx_point = self.dx / n_pt_side
        dy_point = self.dy / n_pt_side
        dz_point = self.dz / n_pt_side
        dxyz_point = np.reshape(
            np.mgrid[
                0:n_pt_side,0:n_pt_side, 0:n_pt_side
            ].T, (total_points, 3)
        ).astype(float)
        for i in range(total_points):
            dxyz_point[i, 0] *= dx_point
            dxyz_point[i, 1] *= dy_point
            dxyz_point[i, 2] *= dz_point
        return dxyz_point

    cdef double get_frac_accessible(
        self, double[:] coords, double[:, ::1] dxyz_point
    ):
        """Get the fraction of a bin which is accessible inside a confinement.

        Parameters
        ----------
        coords : double[:]
            Cartesian coordinates in the form (x, y, z) identifying the center
            of the bin
        dxyz_point : double[:, ::1]
            Array of subgrid points (relative to origin) for determination of
            voxel accessible volume. Rows indicate individual points, and 
            columns indicate x, y, z coordinates for each point.

        Returns
        -------
        double
            Fraction of the bin which is accessible inside the confinement
        """
        cdef long i, n_points
        cdef double n_points_dbl, count_true, frac_accessible
        cdef double[:] dists
        cdef double[:, ::1] points

        n_points = dxyz_point.shape[0]
        n_points_dbl = float(n_points)
        points = dxyz_point.copy()
        for i in range(n_points):
            points[i, 0] += coords[0] - self.dx/2
            points[i, 1] += coords[1] - self.dy/2
            points[i, 2] += coords[2] - self.dz/2
        dists = np.linalg.norm(points, axis=1)
        count_true = 0.
        for i in range(n_points):
            if dists[i] < self.confine_length:
                count_true += 1
        frac_accessible = count_true / n_points_dbl
        return frac_accessible

    def to_file(self, path):
        """Save Field description and Polymer/Mark names to CSV as Series.

        Parameters
        ----------
        path : str
            File path at which to save the CSV file representing the field

        Returns
        -------
        None or str
            Returns `None` if `path` is not `None`; otherwise returns the CSV
            representing the field as a string
        """
        rows = {name: self.dict_[name] for name in self._field_descriptors}
        for i, polymer in enumerate(self.polymers):
            rows[polymer.name] = 'polymer'
        for i, mark in self.marks.iterrows():
            # careful! mark.name is the Series.name attribute
            rows[mark['name']] = 'mark'
        # prints just key,value for each key in rows
        return pd.Series(rows).to_csv(path, header=False)

    @classmethod
    def from_file(cls, path, polymers, marks):
        """Recover field saved with `.to_file()`.

        Notes
        -----
        The zeroths column is an index column, and the first column stores the
        data specifying the field.

        Parameters
        ----------
        path : str
            Path to the CSV file representing the field
        polymers : List[PolymerBase]
            Polymers contained in the field
        marks : pd.DataFrame
            Table describing chemical marks affecting the field

        Returns
        -------
        UniformDensityField
            Discrete density field represented by the CSV file
        """
        field_series = pd.read_csv(path, header=None, index_col=0)[1]
        kwargs = field_series[_field_descriptors].to_dict()
        for key in kwargs.keys():
            if key in _int_field_descriptors:
                kwargs[key] = int(kwargs[key])
            elif key in _float_field_descriptors:
                kwargs[key] = float(kwargs[key])
            elif key in _str_field_descriptors and np.isnan(kwargs[key]):
                kwargs[key] = ""
        polymer_names = field_series[field_series == 'polymer'].index.values
        mark_names = field_series[field_series == 'mark'].index.values
        err_prefix = f"Tried to instantiate class:{cls} from file:{path} with "

        if len(polymers) != len(polymer_names):
            raise ValueError(
                err_prefix + f"{len(polymers)} polymers, but "
                f" there are {len(polymer_names)} listed."
            )
        for polymer in polymers:
            if polymer.name not in polymer_names:
                raise ValueError(
                    err_prefix + f"polymer:{polymer.name}, but "
                    " this polymer was not present in file."
                )
        if len(marks) != len(mark_names):
            raise ValueError(
                err_prefix + f"{len(marks)} marks, but "
                f" there are {len(mark_names)} listed."
            )
        for i, mark in marks.iterrows():
            if mark['name'] not in mark_names:
                raise ValueError(
                    err_prefix + f"mark:{mark}, but "
                    " this mark was not present in file."
                )
        return cls(polymers=polymers, marks=marks, **kwargs)

    def __eq__(self, other):
        """Check if two `UniformDensityField` objects are equivalent.

        Parameters
        ----------
        other : UniformDensityField
            Field being compared to the current field.

        Returns
        -------
        bool
            True if two fields are equivalent, else false.
        """
        return np.all(
            [
                self.polymers == other.polymers,
                self.marks.equals(other.marks),
                self.nx == other.nx,
                self.x_width == other.x_width,
                self.ny == other.ny,
                self.y_width == other.y_width,
                self.nz == other.nz,
                self.z_width == other.z_width
            ]
        )

    @staticmethod
    def _get_corner_bin_index(nx, ny, nz) -> np.ndarray:
        """Set up the index array for density bins corners.

        Notes
        -----
        TODO: this should just be a higher-dimensional array to avoid having to
        do any of this math.

        Parameters
        ----------
        nx, ny, nz :  long
            Number of voxels in the x, y, and z direction of the field

        Returns
        -------
        array_like (nx*ny*nz, 8) of long
            For each of the `nx*ny*nz` voxels in the field, stores indices
            identifying eight vertices at the corners of the voxel
        """
        bin_index = np.zeros((nx * ny * nz, 8), dtype=int)
        count = 0
        for index_z in range(nz):
            if index_z == nz - 1:
                index_zp1 = 0
            else:
                index_zp1 = index_z + 1
            for index_y in range(ny):
                if index_y == ny - 1:
                    index_yp1 = 0
                else:
                    index_yp1 = index_y + 1
                for index_x in range(nx):
                    if index_x == nx - 1:
                        index_xp1 = 0
                    else:
                        index_xp1 = index_x + 1
                    # Populate the bin_index array with the 8 corner bins
                    bin_index[count] = [
                        index_x + nx * index_y + nx * ny * index_z,
                        index_xp1 + nx * index_y + nx * ny * index_z,
                        index_x + nx * index_yp1 + nx * ny * index_z,
                        index_xp1 + nx * index_yp1 + nx * ny * index_z,
                        index_x + nx * index_y + nx * ny * index_zp1,
                        index_xp1 + nx * index_y + nx * ny * index_zp1,
                        index_x + nx * index_yp1 + nx * ny * index_zp1,
                        index_xp1 + nx * index_yp1 + nx * ny * index_zp1
                    ]
                    count += 1
        return bin_index

    def __str__(self):
        """Print summary of the UniformDensityField.
        """
        n_poly = len(self.polymers)
        return f"UniformDensityField<nx={self.nx},ny={self.ny},nz={self.nz}," \
               f"npoly={n_poly}>"

    def get_dict(self):
        """Dictionary representation of the `UniformDensityField` object.

        Returns
        -------
        dict
            Dictionary of key attributes representing the field
        """
        return {
            "x_width" : self.x_width,
            "y_width" : self.y_width,
            "z_width" : self.z_width,
            "nx" : self.nx,
            "ny" : self.ny,
            "nz" : self.nz,
            "num_marks" : self.num_marks,
            "n_bins" : self.n_bins,
            "density" : self.density,
            "confine_type" : self.confine_type,
            "confine_length" : self.confine_length,
            "chi" : self.chi
        }

    cdef double compute_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_inds,
        long packet_size
    ):
        """Compute field contribution of energy change for proposed MC move.

        Notes
        -----
        For polymer affected by a move, compare the field energy of the new 
        and old configurations.

        First verify all points lie inside the specified confining boundary.
        If a point lies outside the boundary, return a high energy to reject
        the move and stop field energy calculation. If all beads affected by
        the move lie inside the confinement, do the following:

        Load the current position and states for the polymer. Using this data,
        identify the density of the polymer and marks in each bin in space.
        Repeat this density calculation for the trial polymer and mark
        configuration. Determine the self-interaction energy based on the
        interpolated densities between each bin using the approach described
        by MacPherson et al [1]. For each mark on the chain, add its
        contribution to the field energy based on interactions with like marks.

        References
        ----------
        [1] MacPherson, Q.; Beltran, B.; Spakowitz, A. J. Bottom–up Modeling of
            Chromatin Segregation Due to Epigenetic Modifications. PNAS 2018,
            115 (50), 12739–12744. https://doi.org/10.1073/pnas.1812268115.

        Parameters
        ----------
        poly : `chromo.PolymerBase`
            The polymer which has been moved.
        inds : array_like (N,)
            Indices of beads being moved
        n_inds : long
            Number of beads affected by the move
        packet_size : long
            Number of points to average together when calculating the field
            energy change of a move; done to reduce the computational expense
            of the field energy calculation (at the expense of precision).
            Currently this does nothing. Packet size is a consideration for
            later on if the simulation is too slow. 

        Returns
        -------
        dE : double
            Change in field energy assocaited with the change in polymer 
            configuration
        """
        cdef long i, ind
        cdef long[:] delta_ind_xyz, bin_inds
        cdef double dE_confine, dE

        # Verify that move does not violate the hard confinement
        dE_confine = self.get_confinement_dE(poly, inds, n_inds, trial=1)
        if dE_confine > 0:
            return dE_confine

        # Find changes in polymer density in affected voxels
        bin_inds = self.get_change_in_density(poly, inds, n_inds)

        # Get change in energy based on differences in bead and mark densities
        dE = self.get_dE_marks_and_beads(poly, inds, n_inds, bin_inds)

        return dE

    cdef double get_confinement_dE(
        self, poly.PolymerBase poly, long[:] inds, long n_beads, int trial
    ):
        """Evaluate the energy associated with a confining boundary.

        Notes
        -----
        A large energy is returned if any point lies outside the confinement;
        otherwise, an energy contribution of zero is returned.

        This method is where different confinement types can be defined.

        Parameters
        ----------
        poly : poly.PolymerBase
            The polymer which has been moved.
        inds : long[:]
            Vector of bead indices for which to evaluate confinement
        n_beads : long
            Number of beads for which to evaluate confinement
        trial : int
            Indicator of whether to evaluate confinement on trial positions (1)
            or current positions (0)

        Returns
        -------
        double
            Energy contribution of the confinement; either zero if all points
            lie inside the confinement or some large value ensuring the move
            gets rejected if any point lies outside the confinement
        """
        cdef long i
        cdef double dist

        # No Confinement
        if self.confine_type == "":
            return 0.

        # Spherical confinement
        elif self.confine_type == "Spherical":
            if trial == 1:
                for i in range(n_beads):
                    dist = sqrt(
                        vec_dot3(poly.r_trial[inds[i]], poly.r_trial[inds[i]])
                    )
                    if dist > self.confine_length:
                        return E_HUGE
            elif trial == 0:
                for i in range(n_beads):
                    dist = sqrt(vec_dot3(poly.r[inds[i]], poly.r[inds[i]]))
                    if dist > self.confine_length:
                        return E_HUGE
            else:
                raise ValueError("Trial flag " + str(trial) + " not found.")
            return 0.

        # Confinement type not found
        else:
            raise ValueError("Confinement type " + self.confine_type + " not found.")

    cdef long[:] get_change_in_density(
        self, poly.PolymerBase poly, long[:] inds, long n_inds
    ):
        """Calculate the density in bins affected by the MC move.

        Notes
        -----
        This method takes in a polymer configuration (either current or trial,
        as dectated by the `trial` argument) and updates a field density map.
        The method operates by the following steps:

        (1) Begin by generating weights and indices affected by the MC move.

            - Start by identifying the (0, 0, 0) bins for the affected beads
            and the associated weights.

            - Then generate a weight array for all bins containing affected
            beads.

            - Finally, generate an index vector identifying the affected bins.
        
        (2) Then initialize a density vector and loop through marks and corner
        bin indices to fill in the density vector.

            - Values in arrays are determined element-by-element to improve
            speed once compiled.

        (3) Combine repeated entries in the density array, and return the
        affected densities and bin indices.

        TODO: Replace dict with set (once I figure out how to get that working)

        NOTE: Replacing k-loops with direct statements improves speed by approx
        0.7% -- we opt for clean code instead.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N, ) of long
            Ordered indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move

        Returns
        -------
        long[:]
            Array of unique bin indices affected by the proposed MC move
        """
        cdef long i, j, k, l, m, ind
        cdef double prefactor
        cdef double[:, :, ::1] densities

        cdef set bins_found
        bins_found = set()

        for i in range(n_inds):
            # Shift positions so all positive, apply periodic boundaries
            # Get the lower neighboring bin index
            for j in range(3):
                # Load current and trial configuration of the polymer (current
                # in row 0; trial in row 1)
                self.xyz_with_trial[0, j] = (
                    (poly.r[inds[i], j] + self.half_width_xyz[j]) %
                    self.width_xyz[j]
                ) - self.half_step_xyz[j]
                self.xyz_with_trial[1, j] = (
                    (poly.r_trial[inds[i], j] + self.half_width_xyz[j]) %
                    self.width_xyz[j]
                ) - self.half_step_xyz[j]

                # Get the lower neighboring bin index
                for k in range(2):
                    ind = <long>floor((self.xyz_with_trial[k, j]) / self.dxyz[j])
                    if ind == -1:
                        self.index_xyz_with_trial[k, j] = self.n_xyz_m1[j]
                    else:
                        self.index_xyz_with_trial[k, j] = ind
                
                    # Get weight in the lower bin index
                    self.weight_xyz_with_trial[k, j] =\
                        (1 - (self.xyz_with_trial[k, j] / self.dxyz[j] - ind))

            # Get weights and superindices of eight bins containing beads
            self._generate_weight_vector_with_trial()
            self._generate_index_vector_with_trial()

            # Distribute weights into bins
            # k indicates current (0) or trial (1) configuration
            # l indicates for which of eight bins density is being calculated
            # m indicates polymer (0) or which protein (1 to n_marks)
            for k in range(2):
                for l in range(8):
                    poly.densities_temp[k, 0, l] = (
                        self.wt_vec_with_trial[k, l] /
                        self.access_vols[self.nbr_inds_with_trial[k, l]]
                    )
                    for m in range(1, poly.n_marks_p1):
                        if k == 0:
                            poly.densities_temp[k, m, l] =\
                                poly.densities_temp[k, 0, l] *\
                                poly.states[inds[i], m-1]
                        else:
                            poly.densities_temp[k, m, l] =\
                                poly.densities_temp[k, 0, l] *\
                                poly.states_trial[inds[i], m-1]

            # Combine repeating elements, stored in poly.density_trial
            for k in range(2):
                # Subtract if current configuration
                if k == 0:
                    prefactor = -1.
                else:
                    prefactor = 1.

                # Accumulate densities
                for l in range(8):
                    if self.nbr_inds_with_trial[k, l] in bins_found:
                        for m in range(poly.n_marks_p1):
                            self.density_trial[self.nbr_inds_with_trial[k, l], m] +=\
                                prefactor * poly.densities_temp[k, m, l]
                    else:
                        bins_found.add(self.nbr_inds_with_trial[k, l])
                        for m in range(poly.n_marks_p1):
                            self.density_trial[self.nbr_inds_with_trial[k, l], m] =\
                                prefactor * poly.densities_temp[k, m, l]

        return np.array(list(bins_found))

    cdef void _generate_weight_vector_with_trial(self):
        """Generate weight array for eight bins containing the bead.
        """
        cdef int i
        for i in range(2):
            self.wt_vec_with_trial[i, 0] = (
                self.weight_xyz_with_trial[i,0] *
                self.weight_xyz_with_trial[i,1] *
                self.weight_xyz_with_trial[i,2]
            )
            self.wt_vec_with_trial[i, 1] = (
                (1-self.weight_xyz_with_trial[i,0]) *
                self.weight_xyz_with_trial[i,1] *
                self.weight_xyz_with_trial[i,2]
            )
            self.wt_vec_with_trial[i, 2] = (
                self.weight_xyz_with_trial[i,0] *
                (1-self.weight_xyz_with_trial[i,1]) *
                self.weight_xyz_with_trial[i,2]
            )
            self.wt_vec_with_trial[i, 3] = (
                (1-self.weight_xyz_with_trial[i,0]) *
                (1-self.weight_xyz_with_trial[i,1]) *
                self.weight_xyz_with_trial[i,2]
            )
            self.wt_vec_with_trial[i, 4] = (
                self.weight_xyz_with_trial[i,0] *
                self.weight_xyz_with_trial[i,1] *
                (1-self.weight_xyz_with_trial[i,2])
            )
            self.wt_vec_with_trial[i, 5] = (
                (1-self.weight_xyz_with_trial[i,0]) *
                self.weight_xyz_with_trial[i,1] *
                (1-self.weight_xyz_with_trial[i,2])
            )
            self.wt_vec_with_trial[i, 6] = (
                self.weight_xyz_with_trial[i,0] *
                (1-self.weight_xyz_with_trial[i,1]) *
                (1-self.weight_xyz_with_trial[i,2])
            )
            self.wt_vec_with_trial[i, 7] = (
                (1-self.weight_xyz_with_trial[i,0]) *
                (1-self.weight_xyz_with_trial[i,1]) *
                (1-self.weight_xyz_with_trial[i,2])
            )

    cdef void _generate_index_vector_with_trial(self):
        """Generate vector of eight superindices containing a bead.

        Notes
        -----
        This is a cleaner form of the method, but is ~70% slower:

        for i in range(2):
            for j in range(8):
                ind_x = ((j % 2) + index_xyz[i, 0]) % self.nx
                ind_y = (((int(floor(j/2))) % 2) + index_xyz[i, 1]) % self.ny
                ind_z = (((int(floor(j/4))) % 2) + index_xyz[i, 2]) % self.nz
                self.nbr_inds_with_trial[i, j] = inds_to_super_ind(
                    ind_x, ind_y, ind_z, self.nx, self.ny
                )
        """
        cdef int i
        for i in range(2):
            self.nbr_inds_with_trial[i, 0] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 1] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 2] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 3] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 4] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 5] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 6] = self.inds_xyz_to_super[
                (self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]
            self.nbr_inds_with_trial[i, 7] = self.inds_xyz_to_super[
                (1 + self.index_xyz_with_trial[i, 0]),
                (1 + self.index_xyz_with_trial[i, 1]),
                (1 + self.index_xyz_with_trial[i, 2])
            ]

    cdef double get_dE_marks_and_beads(
        self, poly.PolymerBase poly, long[:] inds, long n_inds, long[:] bin_inds
    ):
        """Get the change in energy associated with reconfiguration of marks.

        Notes
        -----
        The scale of the oligomerization energy is constant for each epigenetic
        mark. While it does not affect runtime too much, this method can be 
        optimized by precomputing the scale.

        TODO: Remove enumerate call -- that carries overhead

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N, ) of long
            Indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move
        bin_inds : long[:]
            Bin indices affected by the MC move either through loss or addition
            of density

        Returns
        -------
        double
            Change in energy associated with reconfiguration of marks based on
            mark interaction energies, as well as nonspecific bead interactions
        """
        cdef long n_bins, i, j, kn_double_bound
        cdef double tot_density_change, dE_marks_beads
        cdef double[:, ::1] delta_rho_squared, rho_change
        cdef double[:, :, ::1] delta_rho_interact_squared
        cdef dict mark_info

        # Change in squared density for each mark
        n_bins = len(bin_inds)
        rho_change = np.zeros((n_bins, poly.n_marks_p1), dtype='d')
        delta_rho_squared = np.zeros((n_bins, poly.n_marks_p1), dtype='d')
        delta_rho_interact_squared = \
            np.zeros((n_bins, poly.n_marks_p1, poly.n_marks_p1), dtype='d')
        for i in range(n_bins):
            for j in range(poly.n_marks_p1):
                rho_change[i, j] = (self.density_trial[bin_inds[i], j])

                delta_rho_squared[i, j] = (
                    self.density[bin_inds[i], j] +
                    self.density_trial[bin_inds[i], j]
                ) ** 2 - self.density[bin_inds[i], j] ** 2

                for k in range(poly.n_marks_p1):
                    delta_rho_interact_squared[i, j, k] = ((
                        self.density[bin_inds[i], j] +
                        self.density_trial[bin_inds[i], j]
                    ) * (
                        self.density[bin_inds[i], k] +
                        self.density_trial[bin_inds[i], k]
                    )) - (
                        self.density[bin_inds[i], j] *
                        self.density[bin_inds[i], k]
                    )

        # Count the number of nucleosomes bound by mark at both histone tails
        self.count_doubly_bound(poly, inds, n_inds, trial=1)
        self.count_doubly_bound(poly, inds, n_inds, trial=0)

        dE_marks_beads = 0
        for i, mark_info in enumerate(self.mark_dict):
            # Calculate total density change
            tot_density_change = 0
            for j in range(n_bins):
                tot_density_change += delta_rho_squared[j, i+1]

            # Oligomerization
            dE_marks_beads +=\
                mark_info['field_energy_prefactor'] * tot_density_change

            # Intranucleosome interaction
            n_double_bound = self.doubly_bound_trial[i] - self.doubly_bound[i]
            dE_marks_beads +=\
                mark_info['interaction_energy_intranucleosome'] * n_double_bound

        # Cross-talk Interaction
        for i, mark_info in enumerate(self.mark_dict):
            for j, next_mark_info in enumerate(self.mark_dict):
                tot_density_change_interact = 0
                for k in range(n_bins):
                    tot_density_change_interact += delta_rho_interact_squared[k, i, j]
                dE_marks_beads += mark_info["cross_talk_field_energy_prefactor"][next_mark_info["name"]] *\
                    tot_density_change_interact

        # Nonspecific bead interaction energy
        dE_marks_beads += self.nonspecific_interact_dE(poly, bin_inds, n_bins)
        return dE_marks_beads

    cdef double nonspecific_interact_dE(
        self, poly.PolymerBase poly, long[:] bin_inds, long n_bins
    ):
        """Get nonspecific interaction energy for an affected polymer segment.

        Notes
        -----
        This method assumes that all beads are of the same volume.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer affected by the current MC move
        bin_inds : long[:]
            Bin indices affected by the MC move either through loss or addition
            of density
        n_bins : long
            Number of bins affected by an MC move

        Returns
        -------
        double
            Change in energy associated with the nonspecific interactions
            between segments of the polymer chain
        """
        cdef double bead_V, nonspecific_dE, access_vol
        cdef double[:, ::1] vol_fracs

        bead_V = poly.beads[0].vol
        vol_fracs = self.get_volume_fractions_with_trial(bead_V, bin_inds, n_bins)

        nonspecific_dE = 0
        for i in range(n_bins):
            access_vol = self.access_vols[bin_inds[i]]
            
            # Trial Volume Fractions
            if vol_fracs[1, i] > 0.5:
                nonspecific_dE += E_HUGE
            else:
                nonspecific_dE += self.chi * access_vol * vol_fracs[1, i] ** 2
            # Current volume fractions
            if vol_fracs[0, i] > 0.5:
                nonspecific_dE -= E_HUGE
            else:
                nonspecific_dE -= self.chi * access_vol * vol_fracs[0, i] ** 2

        return nonspecific_dE

    cdef double[:, ::1] get_volume_fractions_with_trial(
        self, double bead_V, long[:] bin_inds, long n_bins
    ):
        """Calculate the volume frac. of beads in each bin of discretized space.

        Parameters
        ----------
        bead_V : double
            Volume of an individual bead
        bin_inds : long[:]
            Bin indices affected by the MC move either through loss or addition
            of density
        n_bins : long
            Number of bins affected by an MC move

        Returns
        -------
        double[:, ::1]
            Array of bead volume fractions for each bin; first row corresponds
            to current configuration of the polymer, and the second row
            corresponds to the trial configuration of the polymer
        """
        cdef long i, j
        cdef double access_vol, density, change_in_density, bead_to_voxel_vol
        cdef double[:, ::1] vol_fracs = np.empty((2, n_bins), dtype='d')

        for j in range(n_bins):
            access_vol = self.access_vols[bin_inds[j]]
            density = self.density[bin_inds[j], 0]
            change_in_density = self.density_trial[bin_inds[j], 0]
            bead_to_voxel_vol = bead_V / access_vol
            vol_fracs[0, j] = density * bead_to_voxel_vol
            vol_fracs[1, j] = vol_fracs[0, j] + (
                change_in_density * bead_to_voxel_vol
            )
        return vol_fracs

    cdef void count_doubly_bound(
        self, poly.PolymerBase poly, long[:] inds, long n_inds, bint trial
    ):
        """For each epigenetic mark, count the number of doubly bound beads.

        Notes
        -----
        We refer to nucleosomes with two histone tails bound by an epigenetic
        mark as being doubly bound.

        When both histone tails of a nucleosome are bound by the same
        epigenetic mark, there is an additional intranucleosome interaction
        energy contributing to the binding energy.

        This function counts the number of doubly bounded nucleosomes for each
        epigenetic mark.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N,) of long
            Indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move
        trial : bint
            Indicator for whether to count doubly bound beads for the current
            configuration (0) or trial configuration (0)
        """
        cdef long i, j

        # Count doubly-bound beads for current configuration
        if trial == 0:
            for j in range(poly.num_marks):
                self.doubly_bound[j] = 0
                for i in range(n_inds):
                    if poly.states[inds[i], j] == 2:
                        self.doubly_bound[j] += 1

        # Count doubly-bound beads for current configuration
        elif trial == 1:
            for j in range(poly.num_marks):
                self.doubly_bound_trial[j] = 0
                for i in range(n_inds):
                    if poly.states_trial[inds[i], j] == 2:
                        self.doubly_bound_trial[j] += 1

        # Raise error if invalid `trial` flag is passed
        else:
            raise ValueError("Invalid current/trial state indicator.")

    cpdef double compute_E(self, poly.PolymerBase poly):
        """Compute total field energy for the current polymer configuration.
        
        Notes
        -----
        Load the positions and states of the entire polymer. Then determine the
        number density of beads in each bin of the field. Use these densities
        and mark states on each bead to determine an overall field energy.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer object undergoing transformation in field

        Returns
        -------
        double
            Total energy associated with the field
        """
        cdef long n_inds
        cdef double E
        cdef long[:] inds

        n_inds = poly.num_beads
        inds = np.arange(0, n_inds, 1)
        self.update_all_densities(poly, inds, n_inds)
        E = self.get_E_marks_and_beads(poly, inds, n_inds)
        return E

    cdef void update_all_densities(
        self, poly.PolymerBase poly, long[:]& inds, long n_inds
    ):
        """Update the density of the field for a single polymer.
        
        Notes
        -----
        Updates the voxel densities stored in the field object.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer for which densities are calculated
        inds : long[:] by reference
            Array of bead indices for the entire polymer (to avoid re-computing
            this)
        n_inds : long
            Number of beads in the polymer
        """
        cdef double density
        cdef long i, j, l, m, ind
        cdef long[:] superindices

        # Re-initialize all densities
        for i in range(self.n_bins):
            for j in range(poly.num_marks+1):
                self.density[i, j] = 0
        
        # Iterate through beads and add their densities to corresponding bins
        for i in range(n_inds):
            for j in range(3):
                
                # Load current configuration of the polymer
                self.xyz[j] = (
                    (poly.r[inds[i], j] + self.half_width_xyz[j]) %
                    self.width_xyz[j]
                ) - self.half_step_xyz[j]
                
                # Get the lower neighboring bin index
                ind = <long>floor((self.xyz[j]) / self.dxyz[j])
                if ind == -1:
                    self.index_xyz[j] = self.n_xyz_m1[j]
                else:
                    self.index_xyz[j] = ind

                # Get weight in the lower bin index
                self.weight_xyz[j] = (1 - (self.xyz[j] / self.dxyz[j] - ind))
            
            # Get weights and superindices of eight bins containing beads
            self._generate_weight_vector()
            self._generate_index_vector()

            # Distribute weights into bins
            # l indicates for which of eight bins density is being calculated
            # m indicates polymer (0) or which protein (1 to n_marks)
            for l in range(8):
                density = self.wt_vec[l] / self.access_vols[self.nbr_inds[l]]
                self.density[self.nbr_inds[l], 0] += density
                for m in range(1, poly.n_marks_p1):
                    self.density[self.nbr_inds[l], m] += density *\
                        poly.states[inds[i], m-1]

    cpdef void update_all_densities_for_all_polymers(self):
        """Update the density map for every polymer in the field.

        Notes
        -----
        Updates the voxel densities stored in the field object.
        """
        cdef double density
        cdef long h, i, j, l, m, ind, n_marks_p1, n_inds
        cdef long[:] superindices, inds
        cdef poly.PolymerBase poly

        for h in range(self.n_polymers):
            poly = self.polymers[h]
            inds = np.arange(0, poly.num_beads, 1)
            n_inds = poly.num_beads
            n_marks_p1 = poly.n_marks_p1

            # Re-initialize all densities
            for i in range(self.n_bins):
                for j in range(poly.num_marks+1):
                    self.density[i, j] = 0
            
            # Iterate through beads and add densities to corresponding bins
            for i in range(n_inds):
                for j in range(3):
                    
                    # Load current configuration of the polymer
                    self.xyz[j] = (
                        (poly.r[inds[i], j] + self.half_width_xyz[j]) %
                        self.width_xyz[j]
                    ) - self.half_step_xyz[j]
                    
                    # Get the lower neighboring bin index
                    ind = <long>floor((self.xyz[j]) / self.dxyz[j])
                    if ind == -1:
                        self.index_xyz[j] = self.n_xyz_m1[j]
                    else:
                        self.index_xyz[j] = ind

                    # Get weight in the lower bin index
                    self.weight_xyz[j] = (1 - (self.xyz[j] / self.dxyz[j] - ind))
                
                # Get weights and superindices of eight bins containing beads
                self._generate_weight_vector()
                self._generate_index_vector()

                # Distribute weights into bins
                # l indicates for which of 8 bins density is being calculated
                # m indicates polymer (0) or which protein (1 to n_marks)
                for l in range(8):
                    density = self.wt_vec[l] / self.access_vols[self.nbr_inds[l]]
                    self.density[self.nbr_inds[l], 0] += density
                    for m in range(1, n_marks_p1):
                        self.density[self.nbr_inds[l], m] += density *\
                            poly.states[inds[i], m-1]
        
    cdef void _generate_weight_vector(self):
        """Generate weight array for eight bins containing the bead.
        """
        self.wt_vec = np.array([
            self.weight_xyz[0] * self.weight_xyz[1] *
            self.weight_xyz[2],
            (1-self.weight_xyz[0]) * self.weight_xyz[1] *
            self.weight_xyz[2],
            self.weight_xyz[0] * (1-self.weight_xyz[1]) *
            self.weight_xyz[2],
            (1-self.weight_xyz[0]) * (1-self.weight_xyz[1]) *
            self.weight_xyz[2],
            self.weight_xyz[0] * self.weight_xyz[1] *
            (1-self.weight_xyz[2]),
            (1-self.weight_xyz[0]) * self.weight_xyz[1] *
            (1-self.weight_xyz[2]),
            self.weight_xyz[0] * (1-self.weight_xyz[1]) *
            (1-self.weight_xyz[2]),
            (1-self.weight_xyz[0]) * (1-self.weight_xyz[1]) *
            (1-self.weight_xyz[2])
        ])

    cdef void _generate_index_vector(self):
        """Generate vector of eight superindices containing a bead.
        """
        cdef int j
        cdef long ind_x, ind_y, ind_z

        for j in range(8):
            ind_x = ((j % 2) + self.index_xyz[0]) % self.nx
            ind_y = (((int(floor(j/2))) % 2) + self.index_xyz[1]) % self.ny
            ind_z = (((int(floor(j/4))) % 2) + self.index_xyz[2]) % self.nz
            self.nbr_inds[j] = inds_to_super_ind(
                ind_x, ind_y, ind_z, self.nx, self.ny
            )

    cdef double get_E_marks_and_beads(
        self, poly.PolymerBase poly, long[:] inds, long n_inds
    ):
        """Get total energy of polymer associated with configuration and marks.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer for which densities are calculated.
        inds : long[:] by reference
            Array of bead indices for the entire polymer (to avoid re-computing
            this).
        n_inds : long
            Number of beads in the polymer.

        Returns
        -------
        double
            Field energy associated with the polymer
        """
        cdef double E_marks_beads, tot_density
        cdef long i, j, n_double_bound
        cdef dict mark_info

        # Count the number of nucleosomes bound by mark at both histone tails
        self.count_doubly_bound(poly, inds, n_inds, trial=0)

        E_marks_beads = 0
        for i, mark_info in enumerate(self.mark_dict):
            # Calculate the total density
            tot_density = 0
            for j in range(self.n_bins):
                tot_density += self.density[j, i]
            # Oligomerization
            E_marks_beads +=\
                mark_info['field_energy_prefactor'] * tot_density
            # Intranucleosome interaction
            n_double_bound = self.doubly_bound[i]
            E_marks_beads +=\
                mark_info['interaction_energy_intranucleosome'] * n_double_bound

        # Nonspecific bead interaction energy
        E_marks_beads += self.nonspecific_interact_E(poly)
        return E_marks_beads

    cdef double nonspecific_interact_E(self, poly.PolymerBase poly):
        """Get nonspecific interaction energy for the full polymer.

        Notes
        -----
        This method assumes that all beads are of the same volume.

        Parameters
        ----------
        poly : poly.PolymerBase
            Polymer affected by the current MC move

        Returns
        -------
        double
            Absolute energy associated with the nonspecific interactions
            between segments of the polymer chain
        """
        cdef double bead_V, nonspecific_E, access_vol
        cdef double[:] vol_fracs

        bead_V = poly.beads[0].vol
        vol_fracs = self.get_volume_fractions(bead_V)
        nonspecific_E = 0
        for i in range(self.n_bins):
            access_vol = self.access_vols[i]
            if vol_fracs[i] > 0.5:
                nonspecific_E += E_HUGE
            else:
                nonspecific_E += self.chi * access_vol * vol_fracs[i] ** 2
        return nonspecific_E

    cdef double[:] get_volume_fractions(self, double bead_V):
        """Calculate all volume fractions of beads in bins.

        Parameters
        ----------
        bead_V : double
            Volume of an individual bead

        Returns
        -------
        double[:]
            Array of bead volume fractions for each bin
        """
        cdef long j
        cdef double access_vol, density
        cdef double[:] vol_fracs = np.empty((self.n_bins,), dtype='d')

        for j in range(self.n_bins):
            access_vol = self.access_vols[j]
            density = self.density[j, 0]
            vol_fracs[j] = density * bead_V / access_vol
        return vol_fracs

    cdef double[:, ::1] get_coordinates_at_inds(
        self, double[:, ::1]& r, long[:]& inds, long n_inds
    ):
        """
        Get position of specified beads from current polymer configuration.

        Parameters
        ----------
        r : array_like (N, 3) of double by reference
            x, y, z coordinates of every bead in the polymer where rows
            coorespond to individual beads and columns correspond to 
            Cartesian (x, y, z) coordinates
        inds : array_like (M,) of long by reference
            Indices of M beads involved in the move, where `M=n_inds`
        n_inds : long
            Number of bead indices affected by a move

        Returns
        -------
        double[:, ::1]
            Coordinates for specified indicies from current polymer
            configuration where rows coorespond to individual beads and
            columns correspond to Cartesian (x, y, z) coordinates
        """
        cdef long i, j
        cdef double[:, ::1] r_new
        r_new = np.empty((n_inds, 3), dtype='d')
        for i in range(n_inds):
            for j in range(3):
                r_new[i, j] = r[inds[i], j]
        return r_new

    cdef long[:, ::1] get_states_at_inds(
        self, poly.PolymerBase poly, long[:] inds, long n_inds
    ):
        """
        Get binding states of specified beads from current polymer configuration.

        Parameters
        ----------
        poly : poly,PolymerBase
            Polymer object undergoing transformation in field
        inds : array_like (N, ) of long
            Indices of N beads involved in the move
        n_inds : long
            Number of bead indices affected by a move

        Returns
        -------
        long[:, ::1]
            Binding states on polymer at specified bead indices, where rows
            identify individual beads and columns correspond to binding states
            for each epigenetic mark
        """
        cdef long i, j
        cdef long[:, ::1] states
        states = np.empty((n_inds, poly.num_marks), dtype=int)
        for i in range(n_inds):
            for j in range(poly.num_marks):    
                states[i, j] = poly.states[inds[i], j]
        return states


cdef long inds_to_super_ind(
    long ind_x, long ind_y, long ind_z, long nx, long ny
):
    """Determine the super index of a voxel based on three dimensional indices.

    Notes
    -----
    Given the x, y, and z voxel indices and the number of voxels in the x and y
    dimensions, calculate the super-index of a bin.

    Parameters
    ----------
    ind_x, ind_y, ind_z : long
        Voxel index positions in the x, y, and z dimensions
    nx, ny : long
        Number of bins in the x and y directions of Cartesian space

    Returns
    -------
    long
        Super-index position of the voxel
    """
    cdef long super_ind = ind_x + ind_y * nx + ind_z * nx * ny
    return super_ind


cpdef long[:] super_ind_to_inds(long super_ind, long nx, long ny):
    """Calculate the three-dimensional indices from a super-index.
    
    Notes
    -----
    Given the super-index of a voxel and the number of voxels in the x and y
    directions, calculate the three-dimensional indices of the voxel.

    Parameters
    ----------
    super_ind : long
        Super-index of a voxel
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction

    Returns
    -------
    long[:]
        Indices of bin in x, y, and z directions
    """
    cdef long[:] xyz_inds = np.zeros(3, dtype=int)
    xyz_inds[2] = np.floor(super_ind / (nx * ny))
    xyz_inds[1] = np.floor((super_ind - xyz_inds[2] * nx * ny) / nx)
    xyz_inds[0] = super_ind - (xyz_inds[2] * (nx * ny) + xyz_inds[1] * nx)
    return xyz_inds


cpdef dict assign_beads_to_bins(
    double[:, ::1] r_poly, long n_inds, long nx, long ny, long nz,
    double x_width, double y_width, double z_width
):
    """Create a mapping from bin indices to list of associated beads.
    
    Parameters
    ----------
    r_poly : double[:, ::1]
        Array of bead positions
    n_inds : long
        Number of beads - must be equivalent in length to r_poly
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction
    nz : long
        Number of bins in the z direction
    x_width : double
        Widths of space in the x direction
    y_width : double
        Widths of space in the y direction
    z_width : double
        Widths of space in the z direction

    Returns
    -------
    dict
        Mapping of bin indices to associated beads, where keys indicate
        the bin indices and values are lists of beads contained in the bin
    """
    cdef long i, index_x, index_y, index_z, n_bins
    cdef dict bin_map = {}
    cdef double[:] r_i, x_poly_box, y_poly_box, z_poly_box
    cdef double x_poly_box_minus, y_poly_box_minus, z_poly_box_minus
    cdef double dx, dy, dz

    x_poly_box = r_poly[:, 0].copy()
    y_poly_box = r_poly[:, 1].copy()
    z_poly_box = r_poly[:, 2].copy()

    dx = x_width / nx
    dy = y_width / ny
    dz = z_width / nz

    n_bins = nx * ny * nz

    for i in range(n_bins):
        bin_map[i] = []

    for i in range(n_inds):
        x_poly_box_minus = floor((r_poly[i, 0] - 0.5 * dx) / x_width)
        x_poly_box[i] -= 0.5 * dx + x_width * x_poly_box_minus
        y_poly_box_minus = floor((r_poly[i, 1] - 0.5 * dy) / y_width)
        y_poly_box[i] -= 0.5 * dy + y_width * y_poly_box_minus
        z_poly_box_minus = floor((r_poly[i, 2] - 0.5 * dz) / z_width)
        z_poly_box[i] -= 0.5 * dz + z_width * z_poly_box_minus
    
    for i in range(n_inds):
        index_x = <long>floor(nx * x_poly_box[i] / x_width)
        index_y = <long>floor(ny * y_poly_box[i] / y_width)
        index_z = <long>floor(nz * z_poly_box[i] / z_width)
        index_x0y0z0 = index_x + nx * index_y + nx * ny * index_z

        bin_map[index_x0y0z0].append(i)

    return bin_map


cpdef dict get_neighboring_bins(long nx, long ny, long nz):
    """Generate map of bin indices to all immediately neighboring bin indices.

    Parameters
    ----------
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction
    nz : long
        Number of bins in the z direction

    Returns
    -------
    Dict[long, np.ndarray(ndims=1, dtype=long)]
        Dictionary where keys indicate bin indices and values provide a list of
        all immediately neighboring bin indices, including the key.
    """
    cdef long num_bins, i
    cdef dict neighbors

    num_bins = nx * ny * nz
    neighbors = {}
    for i in range(num_bins):
        neighbors[i] = get_neighbors_at_ind(nx, ny, nz, i, num_bins)
    return neighbors


cpdef long[:] get_neighbors_at_ind(
    long nx, long ny, long nz, long ind, long num_bins
):
    """Identify the bin indices immediately neighboring a specific bin index.
    
    Notes
    -----
    This method is no longer in use -- it was determined that the method for 
    applying periodic boundaries was invalid.
    
    TODO: To use this function, fix the method for applying periodic boundaries.

    Parameters
    ----------
    nx : long
        Number of bins in the x direction
    ny : long
        Number of bins in the y direction
    nz : long
        Number of bins in the z direction
    ind : long
        Super-index of bin for which neighbors are desired
    num_bins : long
        Number of bins contained  in the field
    
    Returns
    -------
    np.ndarray(ndims=1, dtype=long)
        Array of nine bin super-indices immediately neighboring and containing
        the bin specified by `ind`
    """
    return np.array([
        (ind - nx - 1), (ind - nx), (ind - nx + 1),
        (ind - 1), ind, (ind + 1),
        (ind + nx - 1), (ind + nx), (ind + nx + 1),

        (ind - (ny-1)*nx - 1), (ind - (ny-1)*nx), (ind - (ny-1)*nx + 1),
        (ind - ny*nx - 1), (ind - ny*nx), (ind - ny*nx+ 1),
        (ind - (ny+1)*nx - 1), (ind - (ny+1)*nx), (ind - (ny+1)*nx + 1),

        (ind + (ny-1)*nx - 1), (ind + (ny-1)*nx), (ind + (ny-1)*nx + 1),
        (ind + ny*nx - 1), (ind + ny*nx), (ind + ny*nx+ 1),
        (ind + (ny+1)*nx - 1), (ind + (ny+1)*nx), (ind + (ny+1)*nx + 1),
    ]) % num_bins


cpdef dict get_blocks(long num_beads, long block_size):
    """Coarse grain individual beads into blocks of linearly adjacent beads.
    
    Notes
    -----
    Used to improve computational tractibility of contact map generation.

    Parameters
    ----------
    num_beads : long
        Number of beads in full-resolution polymer
    block_size : long
        Number of beads to place into a single block in course-grained model

    Return
    ------
    dict[long, long]
        Mapping of each bead ID to its block.
    """
    cdef long i
    cdef dict blocks

    blocks = {}
    for i in range(num_beads):
        blocks[i] = int(floor(i / block_size))

    return blocks

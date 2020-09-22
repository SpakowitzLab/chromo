"""
Field class

Creates a field object that contains parameters for the field calculations
and functions to generate the densities

"""

import numpy as np
from .util import combine_repeat


class FieldBase:
    """
    A discretization of space for computing energies.

    Must be subclassed to be useful.
    """
    def __init__(self):
        self.polymers = []

    def __str__(self):
        return "Field<>"

    def __contains__(self, poly):
        """Check if a polymer is currently set to interact with this field."""
        # No polymers interact with the base field object
        return self.polymers.__contains__(poly)

    def compute_dE(self, poly, ind0, indf, r, t3, t2, states):
        """
        Compute the change in field energy due to a proposed move.

        In order to prevent unnecessary computation, you can omit any
        parameters which where completely unchanged by the move.

        Parameters
        ----------
        poly : `chromo.Polymer`
            The polymer which has been moved.
        ind0 : int
            The first bead index which was moved.
        indf : int
            The last bead index which was moved.
        r : (N, 3) array_like of float, optional
            The proposed new positions of the moved beads. Throughout this
            function, ``N = indf - ind0 + 1`` is the number of moved beads.
        t3 : (N, 3) array_like of float, optional
            The proposed new tangent vectors.
        t2 : (N, 3) array_like of float, optional
            Proposed new material normals.
        states : (N, M) array_like of int, optional
            The proposed new chemical states, where *M* is the number of
            chemical states associated with the given polymer.

        Returns
        -------
        float
            The change in energy caused by the Polymer's movement in this
            field.
        """
        pass


class UniformDensityField(FieldBase):
    def __init__(self, polymers, x_width, nx, y_width, ny, z_width, nz):
        self.polymers = polymers
        self.x_width = x_width
        self.y_width = y_width
        self.z_width = z_width
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = x_width / nx
        self.dy = y_width / ny
        self.dz = z_width / nz
        self.n_bins = nx * ny * nz
        self.vol_bin = x_width * y_width * z_width / self.n_bins
        self.bin_index = UniformDensityField._get_corner_bin_index(
                self.nx, self.ny, self.nz)
        num_epigenmarks = np.array([len(p.epigenmarks) for p in polymers])
        if np.any(num_epigenmarks != num_epigenmarks[0]):
            raise ValueError("All Polymers must have the same number of "
                             "epigenetic marks for now.")
        # one column of density for each state and one for the actual bead
        # density of the polymer
        self.density = np.zeros((self.n_bins, num_epigenmarks[0] + 1), 'd')
        self._recompute_field()  # initialize the density field

    @staticmethod
    def _get_corner_bin_index(nx, ny, nz):
        """Setup the index array for density bins corners."""
        bin_index = np.zeros((nx * ny * nz, 8), 'd')
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
                    bin_index[count, :] = [
                        index_x + nx * index_y + nx * ny * index_z,
                        index_xp1 + nx * index_y + nx * ny * index_z,
                        index_x + nx * index_yp1 + nx * ny * index_z,
                        index_xp1 + nx * index_yp1 + nx * ny * index_z,
                        index_x + nx * index_y + nx * ny * index_zp1,
                        index_xp1 + nx * index_y + nx * ny * index_zp1,
                        index_x + nx * index_yp1 + nx * ny * index_zp1,
                        index_xp1 + nx * index_yp1 + nx * ny * index_zp1]
                    count += 1
        return bin_index

    def __str__(self):
        n_poly = len(self.polymers)
        return "UniformRectField<nx={self.nx},ny={self.ny},nz={self.nz}," \
               "npoly={n_poly}>"

    def _recompute_field(self, check_consistency=False):
        """
        Completely recompute the energy from scratch.

        Re-running this not in __init__ and using *check_consistency* to verify
        that the answer is not too far from the answer that has been built up
        over monte carlo time is one of the best ways to test for consistency
        of the code.
        """
        new_density = self.density.copy()
        for i, poly in enumerate(self.polymers):
            density_poly, index_xyz = self._calc_density(
                    poly.r, poly.states, 0, poly.num_beads)
            new_density[index_xyz, :] += density_poly
        if check_consistency:
            if not np.all(np.isclose(new_density, self.density)):
                raise RuntimeError(f"Recomputing energy of {self} from scratch"
                                   " produced inconsistent results. There is a"
                                   " bug in this code.")

    def compute_dE(self, poly, ind0, indf, r, t3, t2, states):
        # even if the move does not change the states, we cannot ignore them
        # because they're needed to compute the energy at any point along the
        # polymer
        if states is None:
            states = poly.states
        density_poly, index_xyz = self._calc_density(
                poly.r[ind0:indf, :], poly.states, ind0, indf)
        density_poly_trial, index_xyz_trial = self._calc_density(r, states, ind0, indf)
        delta_density_poly_total = np.concatenate((density_poly_trial, -density_poly))
        delta_index_xyz_total = np.concatenate((index_xyz_trial, index_xyz)).astype(int)
        delta_density, delta_index_xyz = combine_repeat(delta_density_poly_total, delta_index_xyz_total)

        delta_energy_epigen = 0
        for i, (_, epi_info) in enumerate(poly.epigenmarks.iterrows()):
            delta_energy_epigen += 0.5 * epi_info.interaction_energy * np.sum(
                (delta_density[:, i + 1] + self.density[delta_index_xyz, i + 1]) ** 2
                - self.density[delta_index_xyz, i + 1] ** 2)
        return delta_energy_epigen

    def _calc_density(self, r_poly, states, ind0, indf):

        num_epigenmark = states.shape[1]

        # Find the (0,0,0) bins for the beads and the associated weights
        x_poly_box = (r_poly[:, 0] - 0.5 * self.dx
                    - self.x_width * np.floor((r_poly[:, 0] - 0.5 * self.dx) / self.x_width))
        index_x = np.floor(self.nx * x_poly_box / self.x_width).astype(int)
        weight_x = 1 - (x_poly_box / self.dx - index_x)

        y_poly_box = (r_poly[:, 1] - 0.5 * self.dy
                    - self.y_width * np.floor((r_poly[:, 1] - 0.5 * self.dy) / self.y_width))
        index_y = np.floor(self.ny * y_poly_box / self.y_width).astype(int)
        weight_y = 1 - (y_poly_box / self.dy - index_y)

        z_poly_box = (r_poly[:, 2] - 0.5 * self.dz
                    - self.z_width * np.floor((r_poly[:, 2] - 0.5 * self.dz) / self.z_width))
        index_z = np.floor(self.nz * z_poly_box / self.z_width).astype(int)
        weight_z = 1 - (z_poly_box / self.dz - index_z)

        index_x0y0z0 = index_x + self.nx * index_y + self.nx * self.ny * index_z

        # Generate the weight array for all of the bins containing the beads

        weight = weight_x * weight_y * weight_z
        weight = np.concatenate((weight, (1 - weight_x) * weight_y * weight_z))
        weight = np.concatenate((weight, weight_x * (1 - weight_y) * weight_z))
        weight = np.concatenate((weight, (1 - weight_x) * (1 - weight_y) * weight_z))
        weight = np.concatenate((weight, weight_x * weight_y * (1 - weight_z)))
        weight = np.concatenate((weight, (1 - weight_x) * weight_y * (1 - weight_z)))
        weight = np.concatenate((weight, weight_x * (1 - weight_y) * (1 - weight_z)))
        weight = np.concatenate((weight, (1 - weight_x) * (1 - weight_y) * (1 - weight_z)))

        index_xyz_total = index_x0y0z0
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 1])).astype(int)
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 2])).astype(int)
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 3])).astype(int)
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 4])).astype(int)
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 5])).astype(int)
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 6])).astype(int)
        index_xyz_total = np.concatenate((index_xyz_total, self.bin_index[index_x0y0z0, 7])).astype(int)

        density_total = np.zeros((8 * (indf - ind0), num_epigenmark + 1), 'd')
        density_total[:, 0] = weight
        for ind_epigen in range(num_epigenmark):
            for ind_corner in range(8):
                ind_corner0 = ind_corner * (indf - ind0)
                ind_cornerf = ind_corner0 + indf - ind0
                density_total[ind_corner0:ind_cornerf, ind_epigen + 1] = \
                    weight[ind_corner0:ind_cornerf] * states[ind0:indf, ind_epigen]

        # Combine repeat entries in the density array
        density, index_xyz = combine_repeat(density_total, index_xyz_total)

        return density, index_xyz

"""
Routine for calculating the density contribution from an array of bead positions

"""

import numpy as np


def calc_density(r_poly, epigen_bind, num_epigenmark, ind0, indf, field):
    """Calculate the mass-density of a subset of polymer beads."""

    # Find the (0,0,0) bins for the beads and the associated weights
    x_poly_box = (r_poly[:, 0] - 0.5 * field.delta_x
                  - field.length_box_x * np.floor((r_poly[:, 0] - 0.5 * field.delta_x) / field.length_box_x))
    index_x = np.floor(field.num_bins_x * x_poly_box / field.length_box_x).astype(int)
    weight_x = 1 - (x_poly_box / field.delta_x - index_x)

    y_poly_box = (r_poly[:, 1] - 0.5 * field.delta_y
                  - field.length_box_y * np.floor((r_poly[:, 1] - 0.5 * field.delta_y) / field.length_box_y))
    index_y = np.floor(field.num_bins_y * y_poly_box / field.length_box_y).astype(int)
    weight_y = 1 - (y_poly_box / field.delta_y - index_y)

    z_poly_box = (r_poly[:, 2] - 0.5 * field.delta_z
                  - field.length_box_z * np.floor((r_poly[:, 2] - 0.5 * field.delta_z) / field.length_box_z))
    index_z = np.floor(field.num_bins_z * z_poly_box / field.length_box_z).astype(int)
    weight_z = 1 - (z_poly_box / field.delta_z - index_z)

    index_x0y0z0 = index_x + field.num_bins_x * index_y + field.num_bins_x * field.num_bins_y * index_z

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
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 1])).astype(int)
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 2])).astype(int)
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 3])).astype(int)
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 4])).astype(int)
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 5])).astype(int)
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 6])).astype(int)
    index_xyz_total = np.concatenate((index_xyz_total, field.bin_index[index_x0y0z0, 7])).astype(int)

    density_total = np.zeros((8 * (indf - ind0), num_epigenmark + 1), 'd')
    density_total[:, 0] = weight
    for ind_epigen in range(num_epigenmark):
        for ind_corner in range(8):
            ind_corner0 = ind_corner * (indf - ind0)
            ind_cornerf = ind_corner0 + indf - ind0
            density_total[ind_corner0:ind_cornerf, ind_epigen + 1] = \
                weight[ind_corner0:ind_cornerf] * epigen_bind[ind0:indf, ind_epigen]

    # Combine repeat entries in the density array
    density, index_xyz = combine_repeat(density_total, index_xyz_total)

    return density, index_xyz


def combine_repeat(a, idx):
    """
    Combine the repeat entries in an array
    
    :param a: 
    :return: a_combine, idx
    """""
    #    a = np.array([[11, 2], [11, 3], [13, 4], [10, 10], [10, 1]])
    #    b = a[np.argsort(a[:, 0])]
    #    grps, idx = np.unique(b[:, 0], return_index=True)
    #    counts = np.add.reduceat(b[:, 1:], idx)
    #    print(np.column_stack((grps, counts)))

    b = idx[np.argsort(idx)]
    grps, idx = np.unique(b, return_index=True)
    counts = np.add.reduceat(a, idx)
    a_with_index = np.column_stack((grps, counts))
    a_combine = a_with_index[:, 1:]
    idx_combine = a_with_index[:, 0].astype(int)

    return a_combine, idx_combine



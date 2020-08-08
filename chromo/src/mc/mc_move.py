"""
Routines for generating Monte Carlo moves of varying types

"""

import numpy as np
import math as math
from mc.calc_density import calc_density


def mc_move(polymer, epigenmark, density, num_epigenmark, num_polymers, mcmove, mc_move_type, field):

    # MC move type 0: Crank-shaft move
    if mc_move_type == 0:
        for i_move in range(mcmove[mc_move_type].num_per_cycle):
            for i_poly in range(num_polymers):
                crank_shaft_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field)
                mcmove[mc_move_type].num_attempt += 1

    # MC move type 1: End pivot move
    elif mc_move_type == 1:
        pass

    # MC move type 2: Slide move
    elif mc_move_type == 2:
        pass

    # MC move type 3: Slide move
    elif mc_move_type == 3:
        pass


    # MC move type 4: Epigenetic protein binding move
    elif mc_move_type == 4:
        pass

    return


def crank_shaft_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field):

    # Select ind0 and indf for the crank-shaft move
    delta_ind = min(np.random.randint(2, mcmove[0].amp_bead), polymer[i_poly].num_beads)
    ind0 = np.random.randint(polymer[i_poly].num_beads - delta_ind + 1)
    indf = ind0 + delta_ind

    # Generate the rotation matrix and vector around the vector between bead ind0 and indf
    rot_angle = mcmove[0].amp_move * (np.random.rand() - 0.5)

    if ind0 == (indf + 1):
        delta_t3 = polymer[i_poly].t3_poly[ind0, :]
    else:
        delta_t3 = polymer[i_poly].r_poly[indf - 1, :] - polymer[i_poly].r_poly[ind0, :]
        delta_t3 /= np.linalg.norm(delta_t3)

    r_ind0 = polymer[i_poly].r_poly[ind0, :]

    rot_matrix = np.zeros((3, 3), 'd')

    rot_matrix[0, 0] = delta_t3[0] ** 2. + (delta_t3[1] ** 2 + delta_t3[2] ** 2) * math.cos(rot_angle)
    rot_matrix[0, 1] = delta_t3[0] * delta_t3[1] * (1 - math.cos(rot_angle)) - delta_t3[2] * math.sin(rot_angle)
    rot_matrix[0, 2] = delta_t3[0] * delta_t3[2] * (1 - math.cos(rot_angle)) + delta_t3[1] * math.sin(rot_angle)

    rot_matrix[1, 0] = delta_t3[0] * delta_t3[1] * (1 - math.cos(rot_angle)) + delta_t3[2] * math.sin(rot_angle)
    rot_matrix[1, 1] = delta_t3[1] ** 2. + (delta_t3[0] ** 2 + delta_t3[2] ** 2) * math.cos(rot_angle)
    rot_matrix[1, 2] = delta_t3[1] * delta_t3[2] * (1 - math.cos(rot_angle)) - delta_t3[0] * math.sin(rot_angle)

    rot_matrix[2, 0] = delta_t3[0] * delta_t3[2] * (1 - math.cos(rot_angle)) - delta_t3[1] * math.sin(rot_angle)
    rot_matrix[2, 1] = delta_t3[1] * delta_t3[2] * (1 - math.cos(rot_angle)) + delta_t3[0] * math.sin(rot_angle)
    rot_matrix[2, 2] = delta_t3[2] ** 2. + (delta_t3[0] ** 2 + delta_t3[1] ** 2) * math.cos(rot_angle)

    rot_vector = np.cross(r_ind0, delta_t3) * math.sin(rot_angle)
    rot_vector[0] += (r_ind0[0] * (1 - delta_t3[0] ** 2)
                      - delta_t3[0]*(r_ind0[1] * delta_t3[1] + r_ind0[2] * delta_t3[2])) * (1 - math.cos(rot_angle))
    rot_vector[1] += (r_ind0[1] * (1 - delta_t3[1] ** 2)
                      - delta_t3[1]*(r_ind0[0] * delta_t3[0] + r_ind0[2] * delta_t3[2])) * (1 - math.cos(rot_angle))
    rot_vector[2] += (r_ind0[2] * (1 - delta_t3[2] ** 2)
                      - delta_t3[2]*(r_ind0[0] * delta_t3[0] + r_ind0[1] * delta_t3[1])) * (1 - math.cos(rot_angle))

    # Generate the trial positions and orientations

    r_poly_trial = np.zeros((indf - ind0, 3), 'd')
    t3_poly_trial = np.zeros((indf - ind0, 3), 'd')
    for i_bead in range(ind0, indf):
        r_poly_trial[i_bead - ind0, :] = rot_vector + np.matmul(rot_matrix, polymer[i_poly].r_poly[i_bead, :])
        t3_poly_trial[i_bead - ind0, :] = np.matmul(rot_matrix, polymer[i_poly].t3_poly[i_bead, :])

    # Calculate the change in energy
    density_poly, index_xyz = calc_density(polymer[i_poly].r_poly[ind0:indf, :], polymer[i_poly].epigen_bind,
                                           num_epigenmark, ind0, indf, field)
    density_poly_trial, index_xyz_trial = calc_density(r_poly_trial, polymer[i_poly].epigen_bind,
                                                       num_epigenmark, ind0, indf, field)
    delta_density_poly_total = np.concatenate((density_poly_trial, -density_poly))
    delta_index_xyz_total = np.concatenate((index_xyz_trial, index_xyz)).astype(int)
    delta_density, delta_index_xyz = combine_repeat(delta_density_poly_total, delta_index_xyz_total)

    delta_energy_epigen = 0
    for i_epigen in range(num_epigenmark):
        delta_energy_epigen += 0.5 * epigenmark[i_epigen].int_energy * np.sum(
            (delta_density[:, i_epigen + 1] + density[delta_index_xyz, i_epigen + 1]) ** 2
            - density[delta_index_xyz, i_epigen + 1] ** 2)

    delta_energy_poly = calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, i_poly, ind0, indf)
    delta_energy = delta_energy_poly + delta_energy_epigen

    # Determine acceptance of trial based on Metropolis criterion
    if np.random.rand() < math.exp(-delta_energy):
        polymer[i_poly].r_poly[ind0:indf, :] = r_poly_trial
        polymer[i_poly].t3_poly[ind0:indf, :] = t3_poly_trial
        density[delta_index_xyz, :] += delta_density
        mcmove[0].num_success += 1


def calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, i_poly, ind0, indf):
    """
    Calculate the change in polymer energy for the trial

    :param r_poly:
    :param r_poly_trial:
    :param t3_poly:
    :param t3_poly_trial:
    :param polymer:
    :param i_poly:
    :param ind0:
    :param indf:
    :return:
    """

    delta_energy_poly = 0

    # Calculate contribution to polymer energy at the ind0 position
    if ind0 != 0:

        delta_r_trial = r_poly_trial[0, :] - polymer[i_poly].r_poly[ind0 - 1, :]
        delta_r_par_trial = np.dot(delta_r_trial, polymer[i_poly].t3_poly[ind0 - 1, :])
        delta_r_perp_trial = delta_r_trial - delta_r_par_trial * polymer[i_poly].t3_poly[ind0 - 1, :]

        delta_r = polymer[i_poly].r_poly[ind0, :] - polymer[i_poly].r_poly[ind0 - 1, :]
        delta_r_par = np.dot(delta_r, polymer[i_poly].t3_poly[ind0 - 1, :])
        delta_r_perp = delta_r - delta_r_par * polymer[i_poly].t3_poly[ind0 - 1, :]

        bend_vec_trial = (t3_poly_trial[0, :] - polymer[i_poly].t3_poly[ind0 - 1, :]
                          - polymer[i_poly].eta * delta_r_perp_trial)
        bend_vec = (polymer[i_poly].t3_poly[ind0, :] - polymer[i_poly].t3_poly[ind0 - 1, :]
                          - polymer[i_poly].eta * delta_r_perp)

        delta_energy_poly += (0.5 * polymer[i_poly].eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                              + 0.5 * polymer[i_poly].eps_par * (delta_r_par_trial - polymer[i_poly].gamma) ** 2
                              + 0.5 * polymer[i_poly].eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
        delta_energy_poly -= (0.5 * polymer[i_poly].eps_bend * np.dot(bend_vec, bend_vec)
                              + 0.5 * polymer[i_poly].eps_par * (delta_r_par - polymer[i_poly].gamma) ** 2
                              + 0.5 * polymer[i_poly].eps_perp * np.dot(delta_r_perp, delta_r_perp))

    # Calculate contribution to polymer energy at the indf position
    if indf != polymer[i_poly].num_beads:

        delta_r_trial = polymer[i_poly].r_poly[indf, :] - r_poly_trial[indf - ind0 - 1, :]
        delta_r_par_trial = np.dot(delta_r_trial, t3_poly_trial[indf - ind0 - 1, :])
        delta_r_perp_trial = delta_r_trial - delta_r_par_trial * t3_poly_trial[indf - ind0 - 1, :]

        delta_r = polymer[i_poly].r_poly[indf, :] - polymer[i_poly].r_poly[indf - 1, :]
        delta_r_par = np.dot(delta_r, polymer[i_poly].t3_poly[indf - 1, :])
        delta_r_perp = delta_r - delta_r_par * polymer[i_poly].t3_poly[indf - 1, :]

        bend_vec_trial = (polymer[i_poly].t3_poly[indf, :] - t3_poly_trial[indf - ind0 - 1, :]
                          - polymer[i_poly].eta * delta_r_perp_trial)
        bend_vec = (polymer[i_poly].t3_poly[indf, :] - polymer[i_poly].t3_poly[indf - 1, :]
                    - polymer[i_poly].eta * delta_r_perp)

        delta_energy_poly += (0.5 * polymer[i_poly].eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                              + 0.5 * polymer[i_poly].eps_par * (delta_r_par_trial - polymer[i_poly].gamma) ** 2
                              + 0.5 * polymer[i_poly].eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
        delta_energy_poly -= (0.5 * polymer[i_poly].eps_bend * np.dot(bend_vec, bend_vec)
                              + 0.5 * polymer[i_poly].eps_par * (delta_r_par - polymer[i_poly].gamma) ** 2
                              + 0.5 * polymer[i_poly].eps_perp * np.dot(delta_r_perp, delta_r_perp))

    return delta_energy_poly


def combine_repeat(a, idx):
    """
    Combine the repeat entries in an array

    :param a: 
    :return: a_combine, idx
    """
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


def end_pivot_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field):
    """
    Randomly rotate segment from one end of polymer chain.
    
    """
    
    # Generate a rotation angle
    rot_angle = mcmove[1].amp_move * (np.random.rand() - 0.5)

    # Determine the number of beads in the chain
    num_beads = polymer[i_poly].num_beads

    # Randomly select whether to rotate the left (side = 0) or right (side = 1) end of the chain
    side = np.random.randint(0,2)

    if side == 0:
        # Select a random bead at the start (left side) of the chain and identify neighboring point
        ind0 = np.random.randint(num_beads)                # Pick a random bead
        r_ind0 = polymer[i_poly].r_poly[ind0, :]          # Isolate coordinates
        ind1 = ind0 + 1                                   # Pick a neighboring bead
        r_ind1 = polymer[i_poly].r_poly[ind1, :]          # Isolate coordinates

    elif side == 1:
        # Select a random bead on the right end of the chain and identify neighboring point
        ind0 = np.random.randint(ind0 + 1, num_beads + 1) # Pick a random bead
        r_ind0 = polymer[i_poly].r_poly[ind0, :]          # Isolate coordinates
        ind1 = ind0 - 1                                   # Pick a neighboring bead
        r_ind1 = polymer[i_poly].r_poly[ind1, :]          # Isolate coordinates

    # Generate translation matrix such that neighboring point is translated to origin
    translate_mat = np.zeros((4,4))
    translate_mat[0, 0] = 1
    translate_mat[1, 1] = 1
    translate_mat[2, 2] = 1
    translate_mat[3, 3] = 1
    translate_mat[0, 3] = -r_ind1[0]
    translate_mat[1, 3] = -r_ind1[1]
    translate_mat[2, 3] = -r_ind1[2]

    # Generate the inverse of the translation mat
    inv_translation_mat = translation_mat
    inv_translation_mat[0, 3] = -translation_mat[0, 3]
    inv_translation_mat[1, 3] = -translation_mat[1, 3]
    inv_translation_mat[2, 3] = -translation_mat[2, 3]

    # Calculate the length of the projections to point ind0 on yz plane.
    proj_len_yz = math.sqrt(r_ind0[2]**2 + r_ind0[1]**2)

    # Generate rotation matrix such that origin to ind0 is rotated onto the xz plane
    rot_mat_x = np.zeros((4,4))
    rot_mat_x[0, 0] = 1
    rot_mat_x[1, 1] = r_ind0[2] / proj_len_yz
    rot_mat_x[1, 2] = -r_ind0[1] / proj_len_yz
    rot_mat_x[2, 1] = r_ind0[1] / proj_len_yz
    rot_mat_x[2, 2] = r_ind0[2] / proj_len_yz
    rot_mat_x[3, 3] = 1

    # Generate the inverse of the x rotation matrix
    inv_rot_mat_x = rot_mat_x
    inv_rot_mat_x[1, 2] = -rot_mat_x[1, 2]
    inv_rot_mat_x[2, 1] = -rot_mat_x[2, 1]

    # Generate rotation matrix around the y-axis such that the origin to ind0 is rotated onto the z-axis
    rot_mat_y = np.zeros((4,4))
    rot_mat_y[0, 0] = proj_len_yz
    rot_mat_y[0, 2] = -r_ind0[0]
    rot_mat_y[1, 1] = 1
    rot_mat_y[2, 0] = r_ind0[0]
    rot_mat_y[2, 2] = proj_len_yz
    rot_mat_y[3, 3] = 1

    # Generate the inverse of the y rotation matrix
    inv_rot_mat_y = rot_mat_y
    inv_rot_mat_y[0, 2] = -rot_mat_y[0, 2]
    inv_rot_mat_y[2, 0] = -rot_mat_y[2, 0]

    # Generate rotation matrix about the z-axis using the specified rotation angle.
    rot_mat_z = np.zeros((4, 4))
    rot_mat_z[0, 0] = math.cos(rot_angle)
    rot_mat_z[0, 1] = -math.sin(rot_angle)
    rot_mat_z[1, 0] = -rot_mat_z[0, 1]
    rot_mat_z[1, 1] = rot_mat_z[0, 0]
    rot_mat_z[2, 2] = 1
    rot_mat_z[3, 3] = 1

    # Generate full rotation matrix
    rot_matrix = np.matmul(inv_translation_mat, \
        np.matmul(inv_rot_mat_x, \
            np.matmul(inv_rot_mat_y, \
                np.matmul(rot_mat_z, \
                    np.matmul(rot_mat_y, \
                        np.matmul(rot_mat_x, translate_mat))))))

    # Generate a matrix of points undergoing rotation
    if side == 0:
        points = np.ones((4, ind0 - 1))
        for i in range(0,3):
            for j in range(0, ind0 - 1):
                points[i, j] = polymer[i_poly].r_poly[j, i]

    # Generate trial positions
    trial_points = np.matmul(rot_matrix, points)

    # 






def slide_move (polymer, ipoly, mcmove, field):
    """
    Random translation of a segment of beads

    """
    
    pass









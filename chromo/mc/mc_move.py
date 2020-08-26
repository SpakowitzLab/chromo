"""
Routines for generating Monte Carlo moves of varying types
"""
import numpy as np
import math as math
from .calc_density import calc_density


def mc_move(polymers, epigenmarks, density, num_epigenmark, num_polymers, mcmove, field):
    mc_move_type = mcmove.id
    # MC move type 0: Crank-shaft move
    if mc_move_type == 0:
        for i_move in range(mcmove.num_per_cycle):
            for poly in polymers:
                crank_shaft_move(poly, epigenmarks, density, mcmove, field)
                mcmove.num_attempt += 1

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


def crank_shaft_move(polymer, epigenmark, density, mcmove, field):

    # Select ind0 and indf for the crank-shaft move
    delta_ind = min(np.random.randint(2, mcmove.amp_bead), polymer.num_beads)
    ind0 = np.random.randint(polymer.num_beads - delta_ind + 1)
    indf = ind0 + delta_ind

    # Generate the rotation matrix and vector around the vector between bead ind0 and indf
    rot_angle = mcmove.amp_move * (np.random.rand() - 0.5)

    if ind0 == (indf + 1):
        delta_t3 = polymer.t_3[ind0, :]
    else:
        delta_t3 = polymer.r[indf - 1, :] - polymer.r[ind0, :]
        delta_t3 /= np.linalg.norm(delta_t3)

    r_ind0 = polymer.r[ind0, :]

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
        r_poly_trial[i_bead - ind0, :] = rot_vector + np.matmul(rot_matrix, polymer.r[i_bead, :])
        t3_poly_trial[i_bead - ind0, :] = np.matmul(rot_matrix, polymer.t_3[i_bead, :])

    # Calculate the change in energy
    density_poly, index_xyz = calc_density(polymer.r[ind0:indf, :],
                                           polymer.states, ind0, indf, field)
    density_poly_trial, index_xyz_trial = calc_density(r_poly_trial,
                                                       polymer.states,
                                                       ind0, indf, field)
    delta_density_poly_total = np.concatenate((density_poly_trial, -density_poly))
    delta_index_xyz_total = np.concatenate((index_xyz_trial, index_xyz)).astype(int)
    delta_density, delta_index_xyz = combine_repeat(delta_density_poly_total, delta_index_xyz_total)

    delta_energy_epigen = 0
    for i, epi_info in enumerate(epigenmark):
        delta_energy_epigen += 0.5 * epi_info.interaction_energy * np.sum(
            (delta_density[:, i + 1] + density[delta_index_xyz, i + 1]) ** 2
            - density[delta_index_xyz, i + 1] ** 2)

    delta_energy_poly = calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, ind0, indf)
    delta_energy = delta_energy_poly + delta_energy_epigen

    # Determine acceptance of trial based on Metropolis criterion
    if np.random.rand() < math.exp(-delta_energy):
        polymer.r[ind0:indf, :] = r_poly_trial
        polymer.t_3[ind0:indf, :] = t3_poly_trial
        density[delta_index_xyz, :] += delta_density
        mcmove.num_success += 1


def calc_delta_energy_poly(r_poly_trial, t3_poly_trial, polymer, ind0, indf):
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

        delta_r_trial = r_poly_trial[0, :] - polymer.r[ind0 - 1, :]
        delta_r_par_trial = np.dot(delta_r_trial, polymer.t_3[ind0 - 1, :])
        delta_r_perp_trial = delta_r_trial - delta_r_par_trial * polymer.t_3[ind0 - 1, :]

        delta_r = polymer.r[ind0, :] - polymer.r[ind0 - 1, :]
        delta_r_par = np.dot(delta_r, polymer.t_3[ind0 - 1, :])
        delta_r_perp = delta_r - delta_r_par * polymer.t_3[ind0 - 1, :]

        bend_vec_trial = (t3_poly_trial[0, :] - polymer.t_3[ind0 - 1, :]
                          - polymer.eta * delta_r_perp_trial)
        bend_vec = (polymer.t_3[ind0, :] - polymer.t_3[ind0 - 1, :]
                          - polymer.eta * delta_r_perp)

        delta_energy_poly += (0.5 * polymer.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                              + 0.5 * polymer.eps_par * (delta_r_par_trial - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
        delta_energy_poly -= (0.5 * polymer.eps_bend * np.dot(bend_vec, bend_vec)
                              + 0.5 * polymer.eps_par * (delta_r_par - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp, delta_r_perp))

    # Calculate contribution to polymer energy at the indf position
    if indf != polymer.num_beads:

        delta_r_trial = polymer.r[indf, :] - r_poly_trial[indf - ind0 - 1, :]
        delta_r_par_trial = np.dot(delta_r_trial, t3_poly_trial[indf - ind0 - 1, :])
        delta_r_perp_trial = delta_r_trial - delta_r_par_trial * t3_poly_trial[indf - ind0 - 1, :]

        delta_r = polymer.r[indf, :] - polymer.r[indf - 1, :]
        delta_r_par = np.dot(delta_r, polymer.t_3[indf - 1, :])
        delta_r_perp = delta_r - delta_r_par * polymer.t_3[indf - 1, :]

        bend_vec_trial = (polymer.t_3[indf, :] - t3_poly_trial[indf - ind0 - 1, :]
                          - polymer.eta * delta_r_perp_trial)
        bend_vec = (polymer.t_3[indf, :] - polymer.t_3[indf - 1, :]
                    - polymer.eta * delta_r_perp)

        delta_energy_poly += (0.5 * polymer.eps_bend * np.dot(bend_vec_trial, bend_vec_trial)
                              + 0.5 * polymer.eps_par * (delta_r_par_trial - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp_trial, delta_r_perp_trial))
        delta_energy_poly -= (0.5 * polymer.eps_bend * np.dot(bend_vec, bend_vec)
                              + 0.5 * polymer.eps_par * (delta_r_par - polymer.gamma) ** 2
                              + 0.5 * polymer.eps_perp * np.dot(delta_r_perp, delta_r_perp))

    return delta_energy_poly


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

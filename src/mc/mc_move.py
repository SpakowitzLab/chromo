"""
Routines for generating Monte Carlo moves of varying types
"""

import random
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


def select_bead_from_left(num_beads, all_beads, exclude_last_bead = True):
    """
    Randomly select a bead in the polymer chain with an exponentially decaying probability based on index distance from the first point.
    
    Parameters
    ----------
    num_beads:              int
                            Number of beads in the polymer chain

    all_beads:              np.array
                            1D vector of all bead indices in the polymer.

    exclude_last_bead:      Boolean (default: True)
                            Set True to exclude the final bead from selection. This is done when rotating the LHS of the polymer.
    
    Returns
    -------
    ind0:                   int
                            Index of a bead selected with exponentially decaying probability with increasing index distance from bead 1

    """

    if exclude_last_bead == True:
        
        # Calculate total of exponential values for normalization
        norm_const = np.sum(np.exp(1 - all_beads[0:len(all_beads)-2]/(num_beads)))
        # Generate a random number for bead selection
        rand_val = random.uniform(0,1)
        # Select a bead based on exponentially decaying probability with distance from the left side of the polymer
        prob = 0
        for i in range(0, len(all_beads) - 1):
            prob += np.exp(1 - all_beads[i]/num_beads) / norm_const
            if rand_val < prob:
                ind0 = i + 1    ## Check that ind0 starts at 1
                break

    else:
        
        # Calculate total of exponential values for normalization
        norm_const = np.sum(np.exp(1 - all_beads[0:len(all_beads)-1]/(num_beads)))
        # Generate a random number for bead selection
        rand_val = random.uniform(0,1)
        # Select a bead based on exponentially decaying probability with distance from the left side of the polymer
        prob. = 0
        for i in range(0, len(all_beads)):
            prob += np.exp(1 - all_beads[i]/num_beads) / norm_const
            if rand_val < prob:
                ind0 = i + 1    ## Check that ind0 starts at 1
                break

    return(ind0)


def select_bead_from_right(num_beads, all_beads, exclude_first_bead = True):
    """
    Randomly select a bead in the polymer chain with an exponentially decaying probability based on index distance from the last point.
    
    Parameters
    ----------
    num_beads:              int
                            Number of beads in the polymer chain

    all_beads:              np.array
                            1D vector of all bead indices in the polymer.

    exclude_first_bead:     Boolean (default: True)
                            Set True to exclude the first bead from selection. This is done when rotating the RHS of the polymer.
    
    Returns
    -------
    ind0:                   int
                            Index of a bead selected with exponentially decaying probability with increasing index distance from final bead

    """

    if exclude_first_bead == True:

        # Calculate total of exponential values for normalization
        norm_const = np.sum(np.exp(1 - (num_beads - all_beads[1:len(all_beads)-1])/(num_beads)))
        # Generate a random number for bead selection
        rand_val = random.uniform(0,1)
        # Select a bead based on exponentially decaying probability with distance from the right side of the polymer
        prob = 0
        for i in range(1, len(all_beads)):
            prob += np.exp(1 - (num_beads - all_beads[i])/(num_beads)) / norm_const
            if rand_val < prob:
                ind0 = i + 1    ## Check that ind0 starts at 1
                break

    else:

        # Calculate total of exponential values for normalization
        norm_const = np.sum(np.exp(1 - (num_beads - all_beads[0:len(all_beads)-1])/(num_beads)))
        # Generate a random number for bead selection
        rand_val = random.uniform(0,1)
        # Select a bead based on exponentially decaying probability with distance from the right side of the polymer
        prob = 0
        for i in range(0, len(all_beads)):
            prob += np.exp(1 - (num_beads - all_beads[i])/(num_beads)) / norm_const
            if rand_val < prob:
                ind0 = i + 1    ## Check that ind0 starts at 1
                break

    return(ind0)


def select_bead_around_index(num_beads, all_beads, ind0):
    """
    Randomly select a bead in the polymer chain with exponentially decaying probability based on index distance from another point.

    Parameters
    ----------
    num_beads:              int
                            Number of beads in the polymer chain

    all_beads:              np.array
                            1D vector of all bead indices in the polymer.

    ind0:                   int
                            Index of first point

    Returns
    -------

    indf:                   int
                            Index of new point selected based on distance from ind0

	"""

    # Calculate total of exponential values for normalization
    norm_const = np.sum(np.exp(1 - abs(all_beads - ind0) / num_beads))
    # Generate a random number for bead selection
    rand_val = random.uniform(0,1)
    # Select a bead based on exponentially decaying probability with distance from the first bead
    prob = 0
    for i in range(0, len(all_beads)):
        prob += np.exp(1-abs(all_beads[i] - ind0) / num_beads) / norm_const
        if rand_val < prob:
            indf = i + 1    ## Check that indf starts at 1
            break

    return(indf)


def arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle):
    """
    Generate a transformation matrix for rotation of angle rot_angle about an arbitrary axis from points r_ind0 to r_ind1.

    Parameters
    ----------
    r_ind0:         (3, 1) np.array
                    1D column vector of (x, y, z) coordinates for the first point forming the axis of rotation

    r_ind1:         (3, 1) np.array
                    1D column vector of (x, y, z) coordinates for the second point forming the axis of rotation

    rot_angle:      float
                    Magnitude of the angle of rotation about arbitrary axis

    Returns
    -------
    rot_matrix:     (4, 4) np.array
                    Homogeneous rotation matrix for rotation about arbitrary axis

    """

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

    return(rot_matrix)


def end_pivot_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field):
    """
    Randomly rotate segment from one end of polymer chain.
    
    """
    
    # Generate a rotation angle
    rot_angle = mcmove[1].amp_move * (np.random.rand() - 0.5)

    # Determine the number of beads in the chain and assign each bead an index
    num_beads = polymer[i_poly].num_beads - 1
    all_beads = np.arange(0, num_beads)

    # Randomly select whether to rotate the left (side = 0) or right (side = 1) end of the chain
    side = np.random.randint(0,2)

    # Select a random bead at the start (left side) of the chain and identify neighboring point.
    if side == 0:
        ind0 = select_bead_from_left(num_beads, all_beads)      # Randomly pick the first bead
        r_ind0 = polymer[i_poly].r_poly[ind0, :]                # Isolate coordinates
        ind1 = ind0 + 1                                         # Identify the neighboring bead
        r_ind1 = polymer[i_poly].r_poly[ind1, :]                # Isolate coordinates

    # Select a random bead on the right end of the chain and identify neighboring point
    elif side == 1:
        ind0 = select_bead_from_right(num_beads, all_beads)     # Randomly pick the first bead
        r_ind0 = polymer[i_poly].r_poly[ind0, :]                # Isolate coordinates
        ind1 = ind0 - 1                                         # Identify the neighboring bead
        r_ind1 = polymer[i_poly].r_poly[ind1, :]                # Isolate coordinates

    # Generate rotation matrix
    rot_matrix = arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle)
    
    # Generate a matrix of points undergoing rotation
    if side == 0:
        # Consider first the case where the left side of the polymer undergoes the rotation
        r_points = np.ones((4, ind0 - 1))
        t3_points = np.ones((4, ind0 - 1))
        for i in range(0, 3):
            for j in range(0, ind0):
                r_points[i, j] = polymer[i_poly].r_poly[j, i]
                t3_points[i, j] = polymer[i_poly].t3_poly[j, i]

    
    elif side == 1:
        # Then consider that the right side of the polymer undergoes rotation
        r_points np.ones((4, num_beads - ind0))
        t3_points np.ones((4, num_beads - ind0))
        for i in range(0, 3):
            for j in range(0, num_beads - ind0):
                r_points[i, j] = polymer[i_poly].r_poly[j + ind0 + 1, i]
                t3_points[i, j] = polymer[i_poly].t3_poly[j + ind0 + 1, i]

    # Generate trial positions
    r_trial_points = np.matmul(rot_matrix, r_points)
    t3_trial_points = np.matmul(rot_matrix, t3_points)

    # Calculate the change in energy

    # Determine acceptance of trial based on Metropolis criterion

    return


def slide_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field):
    """
    Random translation of a segment of beads

    """
    
    # Generate a random translation amplitude
    translation_amp = mcmove[2].amp_move * (np.random.rand())

    # Randomly partition the translation move into x, y, z components
    rand_z = random.uniform(0,1)
    rand_angle = random.uniform(0, 2*math.pi)

    translation_z = translation_amp * rand_z
    translation_y = math.sqrt(1 - translation_z**2) * math.sin(rand_angle)
    translation_x = math.sqrt(1 - translation_z**2) * math.cos(rand_angle)

    # Determine the number of beads in the chain and assign each bead an index
    num_beads = polymer[i_poly].num_beads - 1
    all_beads = np.arange(0, num_beads)

    # Select a random segment of beads
    ind0 = np.random.randint(num_beads + 1)
    indf = select_bead_around_index(num_beads, all_beads, ind0)

    # Generate a translation matrix
    translation_mat = np.zeros((4,4))
    translation_mat[0, 0] = 1
    translation_mat[1, 1] = 1
    translation_mat[2, 2] = 1
    translation_mat[3, 3] = 1
    translation_mat[0, 3] = translation_x
    translation_mat[1, 3] = translation_y
    translation_mat[2, 3] = translation_z

    # Generate a matrix of points undergoing translation
    r_points = no.ones((4, indf - ind0 + 1))
    for i in range(0, 3):
        for j in range(0, indf - ind0 + 1):
            r_points[i, j] = polymer[i_poly].r_poly[j + ind0, i]

    # Generate trial positions
    r_trial_points = np.matmul(translation_mat, r_points)

    # Calculate the change in energy

    # Determine acceptance of trial based on Metropolis criterion

    return


def tangent_rotation_move(polymer, epigenmark, density, num_epigenmark, i_poly, mcmove, field):
    """
    Randomly rotate the tangent vector of a bead in the chain.

    """

    # Generate a rotation angle
    rot_angle = mcmove[3].amp_move * (np.random.rand() - 0.5)

    # Determine the number of beads in the chain and assign each bead an index value
    num_beads = polymer[i_poly].num_beads - 1
    all_beads = np.arange(0, num_beads)

    # Randomly select a bead in the polymer
    ind0 = random.uniform(0, num_beads)

    # Identify the coordinate and tangent of ind0
    r_ind0 = polymer[i_poly].r_poly[ind0, :]
    t3_ind0 = polymer[i_poly].t3_poly[ind0, :]

    # Select an arbitrary rotation axis through the bead
    phi = random.uniform(0, 2*math.pi)
    theta = random.uniform(0, pi)
    r = 1

    # Generate a second point to create the 
    del_x = r * math.sin(theta) * math.cos(phi)
    del_y = r * math.sin(theta) * math.sin(phi)
    del_z = r * math.cos(theta)
    r_ind1 = r_ind0 + np.array(del_x, del_y, del_z)

    # Generate rotation matrix around axis connecting ind0 and ind1
    rot_matrix = arbitrary_axis_rotation(r_ind0, r_ind1, rot_angle)

    # Rotate the tangent vector by the rotation matrix.
    t3_ind0_trial = np.matmul(rot_matrix, t3_ind0)

    # Calculate the change in energy

    # Determine acceptance of trial based on Metropolis criterion

    return



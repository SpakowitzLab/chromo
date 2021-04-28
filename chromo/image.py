r"""
Imaging module - Generate image files from simulations for PyMol imaging

Notes
-----

"""


import numpy as np


def gen_pymol_file(r_poly, meth_seq = np.array([]), hp1_seq = np.array([]), limit_n = False, n_max = 100000,
                   max_method = 'mid_slice', add_nucleus = False, filename='r_poly.pdb', ring=False):
    r"""

    Parameters
    ----------
    r_poly : (num_beads, 3) float
        Conformation of the chain subjected to active-Brownian forces
    filename : str
        File name to write the pdb file
    ring : bool
        Boolean to close the polymer into a ring

    Returns
    -------

    """

    # Open the file
    f = open(filename, 'w')

    atomname1 = "A1"    # Chain atom type
    resname = "SSN"     # Type of residue (UNKnown/Single Stranded Nucleotide)
    chain = "A"         # Chain identifier
    resnum = 1
    numresidues = len(r_poly[:, 0])
    descrip = "Pseudo atom representation of DNA"
    chemicalname = "Body and ribbon spatial coordinates"
    if len(meth_seq) == 0:
        image_meth_seq = False
    else:
        image_meth_seq = True

    # Determine the bead indices to reduce the total represented to n_max if limit_n True

    if n_max >= numresidues:
        limit_n = False

    if limit_n:
        if max_method == 'mid_slice':
            ind_save, connect_left, connect_right = find_ind_mid_slice(r_poly, n_max, ring)
    else:
        ind_save, connect_left, connect_right = find_ind_total(r_poly, ring)

    # Write the preamble to the pymol file

    f.write('HET    %3s  %1s%4d   %5d     %-38s\n' % (resname, chain, resnum, numresidues, descrip))
    f.write('HETNAM     %3s %-50s\n' % (resname, chemicalname))
    f.write('FORMUL  1   %3s    C20 N20 P21\n' % (resname))

    # Write the conformation to the pymol file

    count = 0
    for ind in range(numresidues):
        if ind_save[ind] == 1:
            if image_meth_seq:
                atomname = 'A' + str(int(meth_seq[ind]))
            else:
                atomname = atomname1
            f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                    (count + 1, atomname, resname, chain, r_poly[ind, 0], r_poly[ind, 1], r_poly[ind, 2], 1.00, 1.00))
            count += 1
    numresidues_save = count

    # Add a nucleus bead to the pymol file

    atomname = 'AN'
    chain = 'B'
    r_com = np.mean(r_poly, axis = 0)
    if add_nucleus:
        f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                (count + 1, atomname, resname, chain, r_com[0], r_com[1], r_com[2], 1.00, 1.00))

    # Define the connectivity in the chain

    count = 0
    for ind in range(numresidues):
        if ind_save[ind] == 1:
            if ind == 0:
                ind_left = numresidues_save - 1
            else:
                ind_left = count - 1
            if ind == numresidues - 1:
                ind_right = 0
            else:
                ind_right = count + 1

            if connect_left[ind] == 1 and connect_right[ind] == 1:
                f.write('CONECT%5d%5d%5d\n' % (count + 1, ind_left + 1, ind_right + 1))
            elif connect_left[ind] == 1 and connect_right[ind] == 0:
                f.write('CONECT%5d%5d\n' % (count + 1, ind_left + 1))
            elif connect_left[ind] == 0 and connect_right[ind] == 1:
                f.write('CONECT%5d%5d\n' % (count + 1, ind_right + 1))
#            elif connect_left[ind] == 0 and connect_right[ind] == 0:
#                f.write('CONECT%5d\n' % (count + 1))
            count += 1

    # Close the file
    f.write('END')
    f.close()

    return


def find_ind_mid_slice(r_poly, n_max, ring):
    r"""

    """

    # Find the value of the x-coordinate and sort the beads according to distance from mean
    x_ave = np.mean(r_poly[:, 0])
    delta_x = r_poly[:, 0] - x_ave
    ind_sort = np.argsort(np.abs(delta_x))

    # Select the first n_max beads based on distance from the mean
    ind_save = np.zeros(len(r_poly[:, 0]))
    connect_left = np.zeros(len(r_poly[:, 0]))
    connect_right = np.zeros(len(r_poly[:, 0]))

    for ind in range(n_max):
        ind_save[ind_sort[ind]] = 1

    # Determine whether adjacent beads are saved to define connections
    for ind in range(n_max):
        if (ind_save[ind_sort[ind]] == 1) and (ind_save[ind_sort[ind] - 1] == 1):
            connect_left[ind_sort[ind]] = 1
        if ind_sort[ind] == (len(r_poly[:, 0]) - 1):
            if (ind_save[ind_sort[ind]] == 1) and (ind_save[0] == 1):
                connect_right[ind_sort[ind]] = 1
        else:
            if (ind_save[ind_sort[ind]] == 1) and (ind_save[ind_sort[ind] + 1] == 1):
                connect_right[ind_sort[ind]] = 1

    # Remove the end connections if ring is false
    if not ring:
        connect_left[0] = 0

    if not ring:
        connect_right[-1] = 0

    return ind_save, connect_left, connect_right


def find_ind_total(r_poly, ring):
    r"""

    """

    ind_save = np.ones(len(r_poly[:, 0]))
    connect_left = np.ones(len(r_poly[:, 0]))
    connect_right = np.ones(len(r_poly[:, 0]))

    if ring:
        connect_left[0] = 1
    else:
        connect_left[0] = 0

    if ring:
        connect_right[-1] = 1
    else:
        connect_right[-1] = 0

    return ind_save, connect_left, connect_right

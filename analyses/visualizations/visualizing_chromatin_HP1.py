"""Render Chromatin Configurations

Usage: python visualizing_chromatin <PATH_TO_CSV_FILE> <PATH_TO_PDB_TO_CREATE> <OPTIONAL: CONFINE_RADIUS>
"""

# Built-in modules
import os
import sys

# External modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_PDB(in_path, out_path, confine_radius=None):

    # Load chromatin configuration
    output_path = in_path
    poly_config = pd.read_csv(output_path, header=[0,1], index_col=0, sep=",")
    poly_config.head()
    r_x = poly_config.loc[:, ("r", "x")].to_numpy()
    r_y = poly_config.loc[:, ("r", "y")].to_numpy()
    r_z = poly_config.loc[:, ("r", "z")].to_numpy()
    HP1 = poly_config.loc[:, ("states", "HP1")].to_numpy()


    def get_PDB_string(x, y, z, ind, bound):
        """ Generate a line for a PDB  file corresponding to a single bead.

        Parameters
        ----------
        x : float
            x-position of the bead
        y : float
            y-position of the bead
        z : float
            z-position of the bead
        ind : int
            Index of the bead
        bound : int
            Indicates the binding state of the protein

        Returns
        -------
        str
            String representing the bead in PDB format
        """

        # Atom indicator (cols 1-4)
        str_out = "ATOM"

        # Atom Serial Number (cols 7-11)
        str_out += 2 * " "
        ind_out = f"{ind}"
        len_ind = len(ind_out)
        if len_ind > 5:
            raise ValueError("The size of the index is not supported in PDB format. Please enter an index between 1 and 99999.")
        pad_size = 5 - len_ind
        str_out += " " * pad_size + ind_out

        # Atom Name (cols 13-16)
        str_out += " "
        if bound == 0:
            str_out += "H   "
        elif bound == 1 or bound == 2:
            str_out += "O   "
        elif bound == 10 or bound == 20:
            str_out += "N   "
        elif bound == -1:
            str_out += "S   "
        else:
            str_out += "C   "

        # Alternate Location Indicator (col 17)
        str_out += " "

        # Residue Name (cols 18-20), Chain Identifier (col 22), Residue sequence number (cols 23-26)
        str_out += "MET A   1"

        # x-position (cols 31-38)
        str_out += "    "
        if x > 9999999 or x < -999999:
            raise ValueError("The size of the x-position is not supported in PDB format. Please limit to 7 characters.")
        if x >= 0:
            if x > 999999:
                str_x = '{: 8.3f}'.format(x)
            else:
                str_x = '{: 8.3f}'.format(x)
        else:
            if x < -99999:
                str_x = '{: 8.3f}'.format(x)
            else:
                str_x = '{: 8.3f}'.format(x)
        str_out += str_x

        # y-position (cols 39-46)
        # str_out += " "
        if y > 9999999 or y < -999999:
            raise ValueError("The size of the y-position is not supported in PDB format. Please limit to 7 characters.")
        if y >= 0:
            if y > 999999:
                str_y = '{: 8.3f}'.format(y)
            else:
                str_y = '{: 8.3f}'.format(y)
        else:
            if y < -99999:
                str_y = '{: 8.3f}'.format(y)
            else:
                str_y = '{: 8.3f}'.format(y)
        str_out += str_y

        # z-position (cols 47-54)
        # str_out += " "
        if z > 9999999 or z < -999999:
            raise ValueError("The size of the z-position is not supported in PDB format. Please limit to 7 characters.")
        if z >= 0:
            if z > 999999:
                str_z = '{: 8.3f}'.format(z)
            else:
                str_z = '{: 8.3f}'.format(z)
        else:
            if z < -99999:
                str_z = '{: 8.3f}'.format(z)
            else:
                str_z = '{: 8.3f}'.format(z)
        str_out += str_z
        return str_out


    pdb_path = out_path
    num_beads = len(r_x)
    max_beads_per_file = 99999

    if num_beads > max_beads_per_file:
        num_files = int(np.ceil(num_beads / max_beads_per_file))
        for file_ind in range(1, num_files+1):
            pdb_path_new = pdb_path.split(".")[0] + "_" + str(file_ind) + ".pdb"
            with open(pdb_path_new, 'w') as f:
                
                start_ind = (file_ind - 1) * max_beads_per_file
                
                if file_ind < num_files:
                    for i in range(max_beads_per_file):
                        line = get_PDB_string(r_x[start_ind + i], r_y[start_ind + i], r_z[start_ind + i], i+1, HP1[start_ind + i])
                        f.write(line + "\n")
                else:
                    for i in range(start_ind, num_beads):
                        line = get_PDB_string(r_x[i], r_y[i], r_z[i], i+1-start_ind, HP1[i])
                        f.write(line + "\n")

                # Draw reference points for circle enclosure
                if file_ind == num_files:
                    if confine_radius is not None:
                        line = get_PDB_string(round(float(confine_radius), 2), 0, 0, max_beads_per_file, -1)
                        f.write(line + "\n")
                        line = get_PDB_string(0, round(float(confine_radius), 2), 0, max_beads_per_file, -1)
                        f.write(line + "\n")
                        line = get_PDB_string(-round(float(confine_radius), 2), 0, 0, max_beads_per_file, -1)
                        f.write(line + "\n")
                        line = get_PDB_string(0, -round(float(confine_radius), 2), 0, max_beads_per_file, -1)
                        f.write(line + "\n")

    else:
        with open(pdb_path, 'w') as f:
            for i in range(num_beads):
                line = get_PDB_string(r_x[i], r_y[i], r_z[i], i+1, HP1[i])
                f.write(line + "\n")

            if confine_radius is not None:
                line = get_PDB_string(round(float(confine_radius), 2), 0, 0, num_beads+1, -1)
                f.write(line + "\n")
                line = get_PDB_string(0, round(float(confine_radius), 2), 0, num_beads+1, -1)
                f.write(line + "\n")
                line = get_PDB_string(-round(float(confine_radius), 2), 0, 0, num_beads+1, -1)
                f.write(line + "\n")
                line = get_PDB_string(0, -round(float(confine_radius), 2), 0, num_beads+1, -1)
                f.write(line + "\n")


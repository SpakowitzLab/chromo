"""
Field class

Creates a field object that contains parameters for the field calculations
and functions to generate the densities

"""

import numpy as np


class Field:
    type = "Field conditions"

    def __init__(self, length_box_x, num_bins_x, length_box_y, num_bins_y, length_box_z, num_bins_z):
        self.name = "Field properties"
        self.vol_bin = length_box_x * length_box_y * length_box_z / (num_bins_x * num_bins_y * num_bins_z)
        self.length_box_x = length_box_x
        self.length_box_y = length_box_y
        self.length_box_z = length_box_z
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.num_bins_z = num_bins_z
        self.delta_x = length_box_x / num_bins_x
        self.delta_y = length_box_y / num_bins_y
        self.delta_z = length_box_z / num_bins_z
        self.num_bins_total = num_bins_x * num_bins_y * num_bins_z

        # Setup the index array for density bins
        bin_index = np.zeros((num_bins_x * num_bins_y * num_bins_z, 8), 'd')
        count = 0
        for index_z in range(num_bins_z):
            if index_z == num_bins_z - 1:
                index_zp1 = 0
            else:
                index_zp1 = index_z + 1
            for index_y in range(num_bins_y):
                if index_y == num_bins_y - 1:
                    index_yp1 = 0
                else:
                    index_yp1 = index_y + 1
                for index_x in range(num_bins_x):
                    if index_x == num_bins_x - 1:
                        index_xp1 = 0
                    else:
                        index_xp1 = index_x + 1
                    # Populate the bin_index array with the 8 corner bins
                    bin_index[count, :] = [
                        index_x + num_bins_x * index_y + num_bins_x * num_bins_y * index_z,
                        index_xp1 + num_bins_x * index_y + num_bins_x * num_bins_y * index_z,
                        index_x + num_bins_x * index_yp1 + num_bins_x * num_bins_y * index_z,
                        index_xp1 + num_bins_x * index_yp1 + num_bins_x * num_bins_y * index_z,
                        index_x + num_bins_x * index_y + num_bins_x * num_bins_y * index_zp1,
                        index_xp1 + num_bins_x * index_y + num_bins_x * num_bins_y * index_zp1,
                        index_x + num_bins_x * index_yp1 + num_bins_x * num_bins_y * index_zp1,
                        index_xp1 + num_bins_x * index_yp1 + num_bins_x * num_bins_y * index_zp1]
                    count += 1
        self.bin_index = bin_index

    def __str__(self):
        return f"{self.name} is a Monte Carlo move"

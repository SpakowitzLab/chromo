#!/usr/bin/env python

import os
import sys
import json
import subprocess

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pymol

from visualizing_chromatin_HP1_PRC1 import generate_PDB


### UPDATE THESE VALUES
out_path = "../../output"
polymer_prefix = "Chr_CG"
sim_range_inclusive = (4, 28)
confine_radius = 364.93211973440407
###

all_content = os.listdir(out_path)
all_content_paths = [f"{out_path}/{item}" for item in all_content]
sim_paths = [path for path in all_content_paths if os.path.isdir(path)]
sim_dict = {int(path.split("_")[-1]): path for path in sim_paths}

sims = {key: sim_dict[key] for key in sim_dict.keys() if key <= sim_range_inclusive[1] and key >= sim_range_inclusive[0]}

latest_files = {}
for sim in sims.keys():
    files = os.listdir(sims[sim])
    files = [file for file in files if file.endswith(".csv") and file.startswith(polymer_prefix)]
    file_inds = [int(file.split(".")[0].split("-")[-1]) for file in files]
    latest_files[sim] = f"{sims[sim]}/{files[np.argmax(file_inds)]}"

binders_paths = {key: f"{sims[key]}/binders" for key in sims.keys()}

self_interact = {}
cross_interact = {}
chemical_potential = {}

for sim in binders_paths.keys():
    df_binders = pd.read_csv(binders_paths[sim], index_col="name")

    if df_binders.loc["HP1", "interaction_energy"] != df_binders.loc["PRC1", "interaction_energy"]:
        raise ValueError("Non-uniform interaction energies")
    if df_binders.loc["HP1", "chemical_potential"] != df_binders.loc["PRC1", "chemical_potential"]:
        raise ValueError("Non-uniform interaction energies")
        
    self_interact[sim] = df_binders.loc["HP1", "interaction_energy"]
    cross_interact[sim] = json.loads(df_binders.loc["HP1", "cross_talk_interaction_energy"].replace("'", "\""))["PRC1"]
    chemical_potential[sim] = df_binders.loc["HP1", "chemical_potential"]

all_data = {}
for key in self_interact.keys():
    
    str_self = str(self_interact[key]).replace("-", "m").replace(".", "d")
    str_cross = str(cross_interact[key]).replace("-", "m").replace(".", "d")
    str_cp = str(chemical_potential[key]).replace("-", "m").replace(".", "d")
    pdb_suffix = f"_self_{str_self}_cross_{str_cross}_cp_{str_cp}.pdb"
    pdb_save_suffix = f"_self_{str_self}_cross_{str_cross}_cp_{str_cp}.png"

    all_data[f"sim_{key}"] = [
        sims[key],
        latest_files[key],
        self_interact[key],
        cross_interact[key],
        chemical_potential[key],
        latest_files[key].replace(".csv", pdb_suffix),
        latest_files[key].replace(".csv", pdb_save_suffix)
    ]

parameters = pd.DataFrame.from_dict(
    all_data,
    orient='index',
    columns=[
        "output_dir",
        "latest_snapshot",
        "self_interaction",
        "cross_interaction",
        "chemical_potential",
        "pdb_path",
        "pdb_save_path"
    ]
).sort_index()

parameters_dict = parameters.to_dict("records")

# Make sure `visualizing_chromatin_HP1_PRC1.py` is in the same directory as this notebook!
for i in range(len(parameters_dict)):
    generate_PDB(
        f"{parameters_dict[i]['latest_snapshot']}",
        f"{parameters_dict[i]['pdb_path']}",
        f"{confine_radius}"
    )

for i in range(len(parameters_dict)):
    pymol.cmd.reinitialize()
    pymol.cmd.load(f"{parameters_dict[i]['pdb_path']}")
    pymol.cmd.show_as("spheres", "all")
    pymol.cmd.png(f"{parameters_dict[i]['pdb_save_path']}")
    pymol.cmd.delete("all")

rows = ["m0d4", "m0d3", "m0d2", "m0d1", "0d0"]
columns = ["m0d5", "m0d25", "0d0", "0d25", "0d5"]

column_headers = [col.replace("m", "-").replace("d", ".") for col in columns]
row_headers = [row.replace("m", "-").replace("d", ".") for row in rows]


def row_cp_col_cross(file_path, rows, columns):
    """Get the row and column position of a configuration in an array of chemical potentials and cross-interactions.

    Each row corresponds to a common chemical potential and each column corresponds to a common cross-interaction.
    """
    pdb_file = file_path.split(".")[-2].split("/")[-1]
    meta_data = pdb_file.split("_")[1:]
    
    self = meta_data[meta_data.index("self")+1]
    cross = meta_data[meta_data.index("cross")+1]
    cp = meta_data[meta_data.index("cp")+1]
    
    return (rows.index(cp), columns.index(cross))


fig = plt.figure(figsize=(64,48))
grid = ImageGrid(
    fig, 111,                  
    nrows_ncols=(len(rows), len(columns)),
    axes_pad=0.1,
    share_all=True
)   

for i in range(len(parameters_dict)):
    image_file = f"{parameters_dict[i]['pdb_save_path']}"
    img = cv.imread(image_file)
    row, col = row_cp_col_cross(image_file, rows, columns)
    ind = row * len(columns) + col
    try:
        ax = grid[ind]
        ax.imshow(img)
    except:
        print("Error loading file: " + image_file)

print("Saving Figure")
plt.savefig("../../output/Configuration_Array_CG.png", dpi=300)

pymol.cmd.quit()

"""Generate contact maps from simulation outputs.

Notes
-----
This module provides utility functions and common analyses, but is not
intended to be run independently.

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 6, 2022
"""

# Built-in Modules
import os
import pickle
from typing import Optional, Tuple, Sequence, Union, Dict, List, Set

# Third-party Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom Modules
import chromo.util.rediscretize as rd
import chromo.fields as fd
import analyses.characterizations.inspect_simulations as inspect


Numeric = Union[int, float]

font = {'family': 'serif',
        'weight': 'normal',
        'size': 24}


def scale_cmap(
    cmap: np.ndarray, percentile: float, max_scale: float
) -> np.ndarray:
    """Scale the values in the contact map.

    Parameters
    ----------
    cmap : np.ndarray (N, N) of float
        Matrix of pairwise contact frequencies (typically log-transformed)
    percentile : float
        The percentile value (between 0 and 100) in contact frequency in
        reference to which the cmap values will be scaled
    max_scale: Optional[float]
        Maximum for scaling the color intensity
    """
    upper_percentile_signals = np.percentile(cmap, percentile)
    cmap *= max_scale / upper_percentile_signals
    return cmap


def plot_cmap(
    cmap: np.ndarray, save_path: str, genomic_interval: [Tuple[int, int]],
    axis_ticks: Sequence[Numeric], axis_label: Optional[str] = "Chr 16 (Mb)",
    max_scale: Optional[float] = None, percentile: Optional[float] = 99.
):
    """Plot a contact map.

    Parameters
    ----------
    cmap : np.ndarray (N, N) of float
        Matrix of pairwise contact frequencies (typically log-transformed)
    save_path : str
        Path at which to save contact map
    genomic_interval : Tuple[int, int]
        Lower and upper bounds for labeling genomic interval of contact matrix
        in order (lower_bound, upper_bound)
    axis_ticks : Sequence[Numeric]
        Tick labels on axes (should be inside the genomic interval)
    axis_label : Optional[str]
        Label for axes of contact map (default = "Chr 16 (Mb)")
    max_scale: Optional[float]
        Maximum for scaling the color intensity (default = None); if none,
        color intensities will be scaled to the maximum value in the contact
        map
    percentile : Optional[float]
        The percentile value (between 0 and 100) in contact frequency in
        reference to which the cmap values will be scaled if `max_scale` is not
        None (default = 99)
    """
    if max_scale is None:
        max_scale = np.ceil(np.max(cmap))
    else:
        cmap = scale_cmap(cmap, percentile, max_scale)
    plt.rc('font', **font)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=600)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.5)
    extents = [
        genomic_interval[0], genomic_interval[1],
        genomic_interval[1], genomic_interval[0]
    ]
    im = ax.imshow(cmap, cmap="Reds", extent=extents, vmin=0, vmax=max_scale)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ticks = np.arange(max_scale+1)
    boundaries = np.linspace(0, max_scale, 1000)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    fig.colorbar(
        im, cax=cax, orientation='vertical', ticks=ticks, boundaries=boundaries
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


def get_approx_cmap(
    output_dir: str, sim_ind: int, cg_factor: int,
    resize_factor: float, num_equilibration: int,
    *,
    sim_prefix: Optional[str] = "sim_",
    polymer_prefix: Optional[str] = "Chr",
    field_file: Optional[str] = "UniformDensityField",
    specific_snap: Optional[bool] = False,
    specific_snap_ind: Optional[int] = 0
) -> np.ndarray:
    """Get approx. contact map with coarse-grained beads and voxel densities.

    Parameters
    ----------
    output_dir : str
        Path to folder containing all simulation output directories
    sim_ind : int
        Integer identifier for the simulation of interest
    cg_factor : int
        For numerical efficiency while approximating contact frequencies,
        we coarse-grain beads of the polymer by grouping sets of beads into
        "super-beads." The position of each super-bead is defined by the
        centroid of its constituent beads. The `cg_factor` specifies how
        many beads to group into each super-bead. A `cg_factor` of 100 or
        50 works well for refined polymer configurations, and a `cg_factor`
        of 5 works well for 15:1 coarse-grained polymer configurations.
    resize_factor : float
        We resize the voxels to improve computational efficiency while
        evaluating contact frequencies. The `resize_factor` specifies the
        approximate factor with which to scale each voxel dimension. A
        `resize_factor` of 7.5 works well for refined polymer configurations,
        and a `resize_factor` of 3 works well for 15:1 coarse-grained polymer
        configurations.
    num_equilibration : int
        Number of equilibration snapshots to exclude from snapshot paths
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
    field_file : Optional[str]
        Name of file in all simulation output directories containing
        the physical parameters of the field, as produced by the
        `to_file()` method of the `UniformDensityField` class
        (default = "UniformDensityField")
    specific_snap : Optional[bool]
        Flag indicating whether or not to generate a contact map from a
        specific snapshot; if `specific_snap` is True, then
        `num_equilibration` has no effect (default = False)
    specific_snap_ind : Optional[int]
        If generating the contact map from a specific snapshot (i.e., if
        `specific_snap` is True), `specific_snap_ind` specifies for which
        snapshot the contact map is to be generated (default = 0); if
        `specific_snap` is False, this parameter has no effect.

    Returns
    -------
    np.ndarray (N, N) of float
        Approximate contact frequencies between pairwise loci; approximated
        using voxels
    """
    sim_names = [f"{sim_prefix}{sim_ind}"]
    sim_paths = {sim_names[0]: f"{output_dir}/{sim_prefix}{sim_ind}"}
    field_dict = inspect.get_field_parameters(
        sim_names, sim_paths, field_file
    )[sim_names[0]]
    x_width = field_dict["x_width"]
    y_width = field_dict["y_width"]
    z_width = field_dict["z_width"]
    nx = round(field_dict["nx"] / resize_factor)
    ny = round(field_dict["ny"] / resize_factor)
    nz = round(field_dict["nz"] / resize_factor)
    n_bins = nx * ny * nz
    if not specific_snap:
        snapshots_filtered = inspect.get_snapshot_paths(
            output_dir, sim_ind, num_equilibration, polymer_prefix,
            sim_prefix
        )
    else:
        snapshots_filtered = [inspect.get_specific_snapshot_path(
            output_dir, sim_ind, specific_snap_ind, polymer_prefix, sim_prefix
        )]
    n_snapshots = len(snapshots_filtered)
    weight = 1 / n_snapshots * cg_factor
    for i, snap in enumerate(snapshots_filtered):
        r = pd.read_csv(
            snap, usecols=[1, 2, 3], skiprows=[0, 1], header=None
        ).to_numpy()
        if i == 0:
            n_beads_full_res = len(r)
            group_intervals = rd.get_cg_bead_intervals(
                n_beads_full_res, cg_factor
            )
            n_beads_cg = len(group_intervals)
            contact_map = np.zeros((n_beads_cg, n_beads_cg), dtype=float)
            nbr_bins = fd.get_neighboring_bins(nx, ny, nz)
        r_cg = rd.get_avg_in_intervals(r, group_intervals)
        bin_map = fd.assign_beads_to_bins(
            np.ascontiguousarray(r_cg), n_beads_cg, nx, ny, nz, x_width,
            y_width, z_width
        )
        for j in range(n_bins):
            for k in nbr_bins[j]:
                if k < j:   # Do not double-count neighbors
                    continue
                for ind0 in bin_map[j]:
                    for ind1 in bin_map[k]:
                        contact_map[ind0, ind1] += weight
                        contact_map[ind1, ind0] += weight
    return contact_map


def plot_approx_cmap(
    output_dir: str, sim_ind: int, cg_factor: int,
    resize_factor: float, num_equilibration: int,
    *,
    sim_prefix: Optional[str] = "sim_",
    polymer_prefix: Optional[str] = "Chr",
    field_file: Optional[str] = "UniformDensityField",
    specific_snap: Optional[bool] = False,
    specific_snap_ind: Optional[int] = 0
):
    """Save approximate contact maps for specified sim in `output` directory.

    Parameters
    ----------
    See documentation for `get_approx_cmap` function.
    """
    save_file = "cmap_approx.csv"
    full_cmap_img = "cmap_approx.png"
    macpherson_cmap_img = "cmap_approx_MacPherson.png"
    contact_map = get_approx_cmap(
        output_dir=output_dir, sim_ind=sim_ind, cg_factor=cg_factor,
        resize_factor=resize_factor, num_equilibration=num_equilibration,
        sim_prefix=sim_prefix, polymer_prefix=polymer_prefix,
        field_file=field_file, specific_snap=specific_snap,
        specific_snap_ind=specific_snap_ind
    )
    log_contacts = np.log10(contact_map+1)
    np.savetxt(
        f"{output_dir}/{sim_prefix}{sim_ind}/{save_file}",
        contact_map,
        delimiter=","
    )
    plot_cmap(
        cmap=log_contacts,
        save_path=f"{output_dir}/{sim_prefix}{sim_ind}/{full_cmap_img}",
        genomic_interval=(0, 90),
        axis_ticks=[15, 30, 45, 60, 75]
    )
    max_ind = len(log_contacts)-1
    lower_bound = int(round((5 / 90) * max_ind))
    upper_bound = int(round((35 / 90) * max_ind))
    plot_cmap(
        cmap=log_contacts[lower_bound:upper_bound, lower_bound:upper_bound],
        save_path=f"{output_dir}/{sim_prefix}{sim_ind}/{macpherson_cmap_img}",
        genomic_interval=(5, 35),
        axis_ticks=[5, 10, 15, 20, 25, 30, 35]
    )


def get_neighbors(
    r: np.ndarray, num_beads: int, cutoff_dist: float,
    nbr_bins: Dict[int, List[int]], nx: int, ny: int, nz: int,
    x_width: float, y_width: float, z_width: float
) -> Dict[int, List[int]]:
    """Generate graph of neighboring nucleosomes.

    Parameters
    ----------
    r : np.ndarray (N, 3) of float
        Cartesian coordinate of each bead in the chromatin fiber;
        rows represent individual beads; columns indicate (x, y, z)
        coordinates
    num_beads : int
        Number of beads representing the chromatin fiber
    cutoff_dist : float
        Cutoff distance between beads below which constitutes a
        near neighbor
    nbr_bins : Dict[int, List[int]]
        Mapping of voxel index to adjacent voxel indices,
    nx, ny, nz : int
        Number of voxels in the x, y, and z directions, respectively
    x_width, y_width, z_width : float
        Dimensions of the voxels in the x, y, and z directions

    Returns
    -------
    Dict[int, List[int]]
        Mapping of each bead index (key) to a list of neighboring
        bead indices (values)
    """
    neighbors = {i: set() for i in range(num_beads)}
    beads_in_bins = fd.assign_beads_to_bins(
        np.ascontiguousarray(r), num_beads,
        nx, ny, nz, x_width, y_width, z_width
    )
    for bin_1 in beads_in_bins.keys():
        for bin_2 in nbr_bins[bin_1]:
            for bead_1 in beads_in_bins[bin_2]:
                for bead_2 in beads_in_bins[bin_2]:
                    dist = np.linalg.norm(r[bead_1] - r[bead_2])
                    if dist < cutoff_dist:
                        if bead_2 not in neighbors[bead_1]:
                            neighbors[bead_1].add(bead_2)
                        if bead_1 not in neighbors[bead_2]:
                            neighbors[bead_2].add(bead_1)
    return neighbors


def get_neighbor_graph(
    output_dir: str,
    sim_ind: int,
    num_equilibration: int,
    nbr_cutoff: float,
    *,
    sim_prefix: Optional[str] = "sim_",
    polymer_prefix: Optional[str] = "Chr",
    field_file: Optional[str] = "UniformDensityField",
    nbr_pickle_file: Optional[str] = "nbr_graph.pkl",
    overwrite_neighbor_graph: Optional[bool] = False,
    specific_snap: Optional[bool] = False,
    specific_snap_ind: Optional[int] = 0
) -> Tuple[int, Dict[str, Dict[int, Set[int]]]]:
    """Generate a contact map from explicit pairwise contacts.
    
    Notes
    -----
    It is too computationally expensive to explicitly calculate pairwise distances
    between beads. Instead, we need to subset beads to voxels larger than the
    contact cutoff distance. Then, for all beads within neighboring voxels, we can
    check pairwise distances and if they fall within the cutoff, we can assign
    those beads as neighbors.

    All beads falling within `nbr_cutoff` of one-another need to be in
    neighboring voxels. To guarentee this is the case, the voxel width should be at
    least `nbr_cutoff`.
    
    To avoid reproducing the neighbor graph from scratch, we will save the neighbor
    list to a pickle file. The pickle file can be "de-pickled" in a later Python
    session to load the neighbor graph.

    To save the neighbor graph to a pickle file, run the code block that calls
    `pickle.dump`. To load the neighbor graph, run the code block that calls
    `pickle.load`.

    Parameters
    ----------
    output_dir : str
        Path to folder containing all simulation output directories
    sim_ind : int
        Integer identifier for the simulation of interest
    num_equilibration : int
        Number of equilibration snapshots to exclude from snapshot paths
    nbr_cutoff : float
        Cutoff separation distance below which a contact exists between two
        beads
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
    field_file : Optional[str]
        Name of file in all simulation output directories containing
        the physical parameters of the field, as produced by the
        `to_file()` method of the `UniformDensityField` class
        (default = "UniformDensityField")
    nbr_pickle_file : Optional[str]
        A near-neighbor graph encoding pairwise contacts is expressed as a
        linked list in the form of a dictionary, where the keys indicate a bead
        index and values indicate contacting beads; when this neighbor graph
        is evaluated, it is stored in a Pickle object to avoid redundant
        calculations; `nbr_pickle_file` gives the filename at which to store
        the neighbor graph Pickle object; the neighbor graph Pickle object will
        be stored to the output directory of the simulation being evaluated
        (default = "nbr_graph.pkl")
    overwrite_neighbor_graph : Optional[bool]
        Flag indicating whether to re-evaluate contacts and overwrite the
        pickled neighbor graph, if it exists (default = False)
    specific_snap : Optional[bool]
        Flag indicating whether or not to generate a contact map from a
        specific snapshot; if `specific_snap` is True, then
        `num_equilibration` has no effect (default = False)
    specific_snap_ind : Optional[int]
        If generating the contact map from a specific snapshot (i.e., if
        `specific_snap` is True), `specific_snap_ind` specifies for which
        snapshot the contact map is to be generated (default = 0); if
        `specific_snap` is False, this parameter has no effect.

    Returns
    -------
    snapshots : List[str]
        List of paths to filtered configurational snapshots
    num_beads : int
        Number of beads in the polymer
    neighbor_graphs : Dict[str, Dict[int], Set[int]]
        Linked lists identifying a graph of contacts for each bead; the outer
        layer dictionary specifies the simulation snapshot; the inner layer
        dictionary provides a linked list, where keys indicate bead indices and
        values indicate a set of contacting beads
    """
    sim_names = [f"{sim_prefix}{sim_ind}"]
    sim_paths = {sim_names[0]: f"{output_dir}/{sim_prefix}{sim_ind}"}
    nbr_pickle_path = f"{sim_paths[sim_names[0]]}/{nbr_pickle_file}"
    neighbor_graph_found = os.path.exists(nbr_pickle_path)
    if not specific_snap:
        snapshots = inspect.get_snapshot_paths(
            output_dir, sim_ind, num_equilibration, polymer_prefix,
            sim_prefix
        )
    else:
        snapshots = [inspect.get_specific_snapshot_path(
            output_dir, sim_ind, specific_snap_ind, polymer_prefix,
            sim_prefix
        )]
    if neighbor_graph_found and not overwrite_neighbor_graph:
        with open(nbr_pickle_path, 'rb') as f:
            neighbor_graphs = pickle.load(f)
        r_example = pd.read_csv(
            f"{snapshots[0]}", header=[0, 1], index_col=0
        ).iloc[:, :3].values
        num_beads = len(r_example)
    else:
        field_dict = inspect.get_field_parameters(
            sim_names, sim_paths, field_file
        )[sim_names[0]]
        x_width = field_dict["x_width"]
        y_width = field_dict["y_width"]
        z_width = field_dict["z_width"]
        nx = int(round(np.floor(x_width / max(nbr_cutoff, 10))))
        ny = int(round(np.floor(y_width / max(nbr_cutoff, 10))))
        nz = int(round(np.floor(z_width / max(nbr_cutoff, 10))))
        n_bins = nx * ny * nz
        nbr_bins = fd.get_neighboring_bins(nx, ny, nz)
        initialized = False
        neighbor_graphs = {}
        for i, snapshot in enumerate(snapshots):
            r = pd.read_csv(
                snapshot, header=[0, 1], index_col=0
            ).iloc[:, :3].values
            if not initialized:
                num_beads = len(r)
                initialized = True
            neighbors = get_neighbors(
                r, num_beads, nbr_cutoff, nbr_bins, nx, ny, nz,
                x_width, y_width, z_width
            )
            neighbor_graphs[snapshot] = neighbors
        with open(nbr_pickle_path, 'wb') as f:
            pickle.dump(neighbor_graphs, f)
    return snapshots, num_beads, neighbor_graphs


def get_detailed_cmap(
    output_dir: str,
    sim_ind: int,
    num_equilibration: int,
    nbr_cutoff: float,
    *,
    sim_prefix: Optional[str] = "sim_",
    polymer_prefix: Optional[str] = "Chr",
    field_file: Optional[str] = "UniformDensityField",
    nbr_pickle_file: Optional[str] = "nbr_graph.pkl",
    overwrite_neighbor_graph: Optional[bool] = False,
    specific_snap: Optional[bool] = False,
    specific_snap_ind: Optional[int] = 0,
    cmap_num_bins: Optional[int] = 1000
) -> np.ndarray:
    """Generate a contact map from explicit pairwise contacts.
    
    Notes
    -----
    We will generate a contact matrix using the neighbor graph. If the chromatin
    fiber is larger than the dimensions of the contact matrix, we will interpolate
    bead indices into the contact matrix.

    Parameters
    ----------
    output_dir : str
        Path to folder containing all simulation output directories
    sim_ind : int
        Integer identifier for the simulation of interest
    num_equilibration : int
        Number of equilibration snapshots to exclude from snapshot paths
    nbr_cutoff : float
        Cutoff separation distance below which a contact exists between two
        beads
    sim_prefix : Optional[str]
        Prefix of simulation output directory names (default = "sim_")
    polymer_prefix : Optional[str]
        Prefix of files containing polymer configurational snapshots
    field_file : Optional[str]
        Name of file in all simulation output directories containing
        the physical parameters of the field, as produced by the
        `to_file()` method of the `UniformDensityField` class
        (default = "UniformDensityField")
    nbr_pickle_file : Optional[str]
        A near-neighbor graph encoding pairwise contacts is expressed as a
        linked list in the form of a dictionary, where the keys indicate a bead
        index and values indicate contacting beads; when this neighbor graph
        is evaluated, it is stored in a Pickle object to avoid redundant
        calculations; `nbr_pickle_file` gives the filename at which to store
        the neighbor graph Pickle object; the neighbor graph Pickle object will
        be stored to the output directory of the simulation being evaluated
        (default = "nbr_graph.pkl")
    overwrite_neighbor_graph : Optional[bool]
        Flag indicating whether to re-evaluate contacts and overwrite the
        pickled neighbor graph, if it exists (default = False)
    specific_snap : Optional[bool]
        Flag indicating whether or not to generate a contact map from a
        specific snapshot; if `specific_snap` is True, then
        `num_equilibration` has no effect (default = False)
    specific_snap_ind : Optional[int]
        If generating the contact map from a specific snapshot (i.e., if
        `specific_snap` is True), `specific_snap_ind` specifies for which
        snapshot the contact map is to be generated (default = 0); if
        `specific_snap` is False, this parameter has no effect.
    cmap_num_bins : Optional[int]
        Number of bins to use for each dimension in detailed contact map
        (default = 1000)

    Returns
    -------
    np.ndarray (N, N) of float
        Explicit contact frequencies between pairwise loci; calculated from
        pairwise distances
    """
    snapshots, num_beads, neighbor_graphs = get_neighbor_graph(
        output_dir, sim_ind, num_equilibration, nbr_cutoff,
        sim_prefix=sim_prefix, polymer_prefix=polymer_prefix,
        field_file=field_file, nbr_pickle_file=nbr_pickle_file,
        overwrite_neighbor_graph=overwrite_neighbor_graph,
        specific_snap=specific_snap, specific_snap_ind=specific_snap_ind
    )
    supports_full_polymer = (cmap_num_bins >= num_beads)
    if supports_full_polymer:
        cmap_num_bins = num_beads
        cmap_bin_width = 1
    else:
        cmap_bin_width = num_beads / cmap_num_bins
    cmap = np.zeros((cmap_num_bins+1, cmap_num_bins+1))
    for snapshot in snapshots:
        graph = neighbor_graphs[snapshot]
        for bead_0 in graph.keys():
            # Interpolate bead_1 position in contact map
            if supports_full_polymer:
                x0 = bead_0
                x1 = x0+1
                w_x0 = 1
                w_x1 = 0
            else:
                x0 = int(np.floor(bead_0 / cmap_bin_width))
                x1 = x0 + 1
                w_x0 = 1 - ((bead_0 / cmap_bin_width) - x0)
                w_x1 = 1 - w_x0
            for bead_1 in graph[bead_0]:
                # Interpolate bead_2 position in contact map
                if supports_full_polymer:
                    y0 = bead_1
                    y1 = y0+1
                    w_y0 = 1
                    w_y1 = 0
                else:
                    y0 = int(np.floor(bead_1 / cmap_bin_width))
                    y1 = y0+1
                    w_y0 = 1 - ((bead_1 / cmap_bin_width) - y0)
                    w_y1 = 1 - w_y0
                # Bilinear Interpolation Weightings
                w00 = w_x0 * w_y0
                w01 = w_x0 * w_y1
                w10 = w_x1 * w_y0
                w11 = w_x1 * w_y1
                cmap[x0, y0] += w00
                cmap[x0, y1] += w01
                cmap[x1, y0] += w10
                cmap[x1, y1] += w11
    # Contacts are double-counted
    cmap /= 2
    # Average over snapshots
    cmap /= len(neighbor_graphs.keys())
    return cmap


def plot_detailed_cmap(
    output_dir: str,
    sim_ind: int,
    num_equilibration: int,
    nbr_cutoff: float,
    *,
    sim_prefix: Optional[str] = "sim_",
    polymer_prefix: Optional[str] = "Chr",
    field_file: Optional[str] = "UniformDensityField",
    nbr_pickle_file: Optional[str] = "nbr_graph.pkl",
    overwrite_neighbor_graph: Optional[bool] = False,
    specific_snap: Optional[bool] = False,
    specific_snap_ind: Optional[int] = 0,
    cmap_num_bins: Optional[int] = 1000
):
    """Save explicit contact maps for specified sim in `output` directory.

    Parameters
    ----------
    See documentation for `get_detailed_cmap` function.
    """
    save_file = "cmap_detailed.csv"
    full_cmap_img = "cmap_detailed.png"
    macpherson_cmap_img = "cmap_detailed_MacPherson.png"
    contact_map = get_detailed_cmap(
        output_dir, sim_ind, num_equilibration, nbr_cutoff,
        sim_prefix=sim_prefix, polymer_prefix=polymer_prefix,
        field_file=field_file, nbr_pickle_file=nbr_pickle_file,
        overwrite_neighbor_graph=overwrite_neighbor_graph,
        specific_snap=specific_snap, specific_snap_ind=specific_snap_ind,
        cmap_num_bins=cmap_num_bins
    )
    log_contacts = np.log10(contact_map+1)
    np.savetxt(
        f"{output_dir}/{sim_prefix}{sim_ind}/{save_file}",
        contact_map,
        delimiter=","
    )
    plot_cmap(
        cmap=log_contacts,
        save_path=f"{output_dir}/{sim_prefix}{sim_ind}/{full_cmap_img}",
        genomic_interval=(0, 90),
        axis_ticks=[15, 30, 45, 60, 75]
    )
    max_ind = len(log_contacts)-1
    lower_bound = int(round((5 / 90) * max_ind))
    upper_bound = int(round((35 / 90) * max_ind))
    plot_cmap(
        cmap=log_contacts[lower_bound:upper_bound, lower_bound:upper_bound],
        save_path=f"{output_dir}/{sim_prefix}{sim_ind}/{macpherson_cmap_img}",
        genomic_interval=(5, 35),
        axis_ticks=[5, 10, 15, 20, 25, 30, 35]
    )

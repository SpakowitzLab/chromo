from pathlib import Path
import numpy as np


_util_folder = Path(__file__).parent.absolute()
dss_params = np.loadtxt(_util_folder / Path("dssWLCparams"))


def combine_repeat(a, idx):
    """Combine the repeat entries in an array of bin densities.

    Flatten one dimension of the value array `a` so each column represents a
    different epigenetic mark, but densities in all bins are combined into the
    same column.

    Flatten the index array `idx` so that indices for each of the eight bins
    representing the density distribution of a bead are in the same column.

    Sort the index array `idx` and the value array `a` by the values in `idx`

    Identify the unique bins and the indices where they are first encountered
    in the sourted index array.

    Add the density within each bin (separately for each epigenetic mark).

    Parameters
    ----------
    a : double[:, :, ::1]
        3D memoryview representing the distribution of densities for each bead
        (dimension 0) in the nearest eight neighboring bins (dimension 2) for
        each epigenetic mark (dimension 1)
    idx : long[:, ::1]
        2D memoryview of bin indices; the first dimension represents each bead
        affected, and the second dimension contains the indices of the eight
        bins containing the bead

    Returns
    -------
    a_combine : double[:, ::1]
        2D memoryview of total density in each affected bin (dimension 0) of
        the polymer and each epigenetic mark (dimension 1)
    """
    a = np.asarray(a)
    idx = np.asarray(idx)
    if len(a.shape) == 3:
        a = a.transpose(0, 2, 1).reshape(a.shape[0] * a.shape[2], a.shape[1])
    if len(idx.shape) > 1:
        idx = idx.flatten()
    argsort_inds = np.argsort(idx, kind='mergesort')
    sorted_inds = idx[argsort_inds]
    sorted_a = a[argsort_inds, :]
    grps, grp_cutoffs = np.unique(sorted_inds, return_index=True)
    counts_in_grps = np.add.reduceat(sorted_a, grp_cutoffs)
    a_with_index = np.column_stack((grps, counts_in_grps))
    a_combine = np.ascontiguousarray(a_with_index[:, 1:])
    idx_combine = np.ascontiguousarray(a_with_index[:, 0].astype(int))
    return a_combine, idx_combine

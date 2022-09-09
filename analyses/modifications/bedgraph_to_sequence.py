"""Convert bedgraph to nucleosome-scale modification sequence.

Notes
-----
Given a BedGraph file showing "fold-change over control" signals for a
particular epigenetic mark in arbitrary genomic intervals, generate a sequence
of modification patterns for that mark at nucleosome-scale resolution. In this
sequence, values of `0`, `1`, and `2` indicate no histone tail, one histone
tail, or two histone tails expressing the mark, respectively, on the respective
nucleosome.

Author:     Joseph Wakim
Group:      Spakowitz Lab @ Stanford
Date:       September 8, 2022
"""

from typing import Optional

import numpy as np
import pandas as pd
import bioframe as bf


def get_cutoffs(fraction_methylated: float) -> np.ndarray:
    """Get cutoff signal strengths between 0/1 and 1/2 tails methylated.
    
    Notes
    -----
    This function assumes that all nucleosomes have two sites that can be
    modified (one on each histone tail).
    
    Parameters
    ----------
    fraction_methylated : float
        Fraction of histone tails on the chromatin fiber expressing the
        epigenetic mark
        
    Returns
    -------
    np.ndarray (2,) of float
        Cutoff signal strengths between 0/1 and 1/2 tails methylated to
        achieve the targeted overall fraction of histone tails methyalted
    """
    cutoff_width = (2 * fraction_methylated) / 3
    pct_cutoffs = np.array([cutoff_width, 2 * cutoff_width])
    return pct_cutoffs


def read_signals_from_bedgraph(
    path: str,
    schema: Optional[str] = "bed4",
    signal_col: Optional[str] = "value"
) -> pd.DataFrame:
    """Load signals from a bedgraph file.
    
    Parameters
    ----------
    path : str
        Path to the bedgraph file
    schema : Optional[str]
        Schema of bedgraph file (default = "bed4")
    signal_col : Optional[str]
        Name to call the signal column (default = "fold_change")\
    
    Returns
    -------
    pd.DataFrame
        Table of signals within genomic intervals
    """
    df_signals = bf.read_table(path, schema=schema)
    df_signals.columns = df_signals.columns[:-1].tolist() + [signal_col]
    return df_signals


def rediscretize_signals(
    df_signals: pd.DataFrame,
    chrom_name: str,
    bin_size: Optional[int] = 200,
    ref_genome: Optional[str] = "hg38",
    signal_col: Optional[str] = "value",
) -> pd.DataFrame:
    """Rediscretize signals into specified, even-width bins.

    Notes
    -----
    BedGraph files discretize signals into arbitrary bins. This
    function redistributes signals into specified, uniform-width
    bins.

    Start by loading chromosome sizes from the reference genome.
    Define genomic intervals with uniform widths matching `bin_size`.
    Determine the overlap between the original genomic intervals and
    the rediscretized intervals. Scale the original signals with the
    amount of overlap between the original genomic intervals and the
    rediscretized intervals. Group signals corresponding to the same
    rediscretized genomic interval.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Table of signals within genomic intervals
    chrom_name : str
        Chromosome identifier in bedgraph file (e.g., "chr1" often
        identifies chromosome 1)
    bin_size : Optional[int]
        Uniform bin width into which signals will be rediscretized
    ref_genome : Optional[str]
        Name of the reference genome from which chromosome sizes
        will be extracted
    signal_col : Optional[str]
        Column of `df_signals` containing signals to be redistributed

    Returns
    -------
    pd.DataFrame
        Table of rediscretized signals in uniform-width genomic
        intervals
    """
    chromsizes = bf.fetch_chromsizes(ref_genome, as_bed=False)
    chromsizes = chromsizes[:22]
    bins = bf.binnify(chromsizes, binsize=bin_size)
    bins = bins[bins.chrom == chrom_name]
    overlap = bf.overlap(bins, df_signals, return_overlap="True")
    
    def scale_overlapping_signals(
        overlap: pd.DataFrame, value_col: Optional[str] = "value",
        value_col_scaled: Optional[str] = "value_scaled"
    ) -> pd.DataFrame:
        f"""Proportionally split signals in overlapping genomic intervals.

        Notes
        -----
        Distributes signals so that they are split proportionally to the
        width of overlap between the rediscretized bins and the original

        Parameters
        ----------
        overlap : pd.DataFrame
            Dataframe produced from `bf.overlap`, representing signals
            from a bedgraph file copied into rediscretized genomic
            intervals; contains columns for
                (1) `chrom` (chromosome name),
                (2) `start` (starting genomic position of rediscretized
                    bin in the overlap),
                (3) `end` (ending genomic position of rediscretized
                    bin in the overlap),
                (4) `true_start` (true genomic position at which
                    original bin starts overlaps with rediscretized bin)
                (5) `true_end` (true genomic position at which
                    original bin ends overlaps with rediscretized bin),
                (6) `{value_col}_` (signal from original bin, copied into 
                    each rediscretized bin which overlaps original bin)
        value_col: Optional[str]
            Name of the column containing original signal values
            (default = "value_")
        value_col_scaled : Optional[str]
            Name of the column into which the rescaled signal values will
            be stored (default = "value_scaled")

        Returns
        --------
        pd.DataFrame
            Table of rescaled signal values in the rediscretized genomic
            intervals
        """
        new_scale = (
            overlap['True_end'] - overlap['True_start']
        ).to_numpy()
        original_width = (
            overlap['end_'] - overlap['start_']
        ).to_numpy()
        overlap[f"{value_col_scaled}"] = \
            overlap[f"{value_col}_"] / original_width * new_scale
        overlap[f"{value_col_scaled}"] = overlap.groupby(
            ["chrom", "start", "end"]
        )[f"{value_col_scaled}"].transform("sum")
        processed_sigs = overlap.drop_duplicates(
            subset=['chrom', 'start', 'end']
        )
        if "label" in list(overlap):
            processed_sigs = processed_sigs[
                ["chrom", "start", "end", f"{value_col_scaled}", "label"]
            ]
        else:
            processed_sigs = \
                processed_sigs[
                ["chrom", "start", "end", f"{value_col_scaled}"]
            ]
        return processed_sigs
    
    return scale_overlapping_signals(
        overlap, signal_col, f"{signal_col}_scaled"
    )


def check_rediscretization(
    df_original: pd.DataFrame, df_new: pd.DataFrame,
    value_col: Optional[str] = "value",
    value_col_scaled: Optional[str] = "value_scaled"
) -> bool:
    """Verify that total signal is maintained during rediscretization.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        Table of signals within genomic intervals
    df_new : pd.DataFrame
        Table of rediscretized signals in uniform-width genomic
        intervals
    value_col : Optional[str]
        Name of the column containing signals in the original data frame
        (default = "value")
    value_col_scaled : Optional[str]
        Name of the column into which the rescaled signal values will
        be stored (default = "value_scaled")
        
    Returns
    -------
    bool
        Indicator of whether or not the total signal in the original
        and rediscretized data frames are consistent
    """
    sum_signals_original = np.sum(df_original[value_col].to_numpy())
    sum_signals_new = np.sum(df_new[value_col_scaled])
    return np.isclose(sum_signals_original, sum_signals_new)


def get_modification_pattern(
    df_rediscretized: pd.DataFrame,
    fraction_methylated: float,
    pct_cutoffs: np.ndarray,
    value_col_scaled: Optional[str] = "value_scaled",
    max_iters: Optional[int] = 1000,
    tolerance: Optional[float] = 0.001
) -> np.ndarray:
    """Get sequence of modification patterns from signal values.
    
    Parameters
    ----------
    df_rediscretized : pd.DataFrame
        Table of rescaled signal values in the rediscretized genomic
        intervals
    fraction_methylated : float
        Fraction of histone tails on the chromatin fiber expressing the
        epigenetic mark
    pct_cutoffs : np.ndarray (2,) of float
        Cutoff signal strengths between 0/1 and 1/2 tails methylated to
        achieve the targeted overall fraction of histone tails methylated
    value_col_scaled : Optional[str]
        Name of the column into which the rescaled signal values will
        be stored (default = "value_scaled")
    max_iters : Optional[int]
        Maximum number of iterations to run while converging on
        modification pattern matching the desired fraction modified
    tolerance : Optional[float]
        Tolerable deviation from the requested fraction methylated
        
    Returns
    -------
    np.ndarray (N,) of int
        Sequence of modification states corresponding to the signal
        strengths; elements correspond to individual nucleosomes;
        values of 0, 1, or 2 indicate the number of histone tails
        modified on the respective nucleosome
    """
    num_beads = len(df_rediscretized)
    value_scaled = df_rediscretized[value_col_scaled].to_numpy().flatten()
    avg_value = np.average(value_scaled)
    cutoffs = np.percentile(value_scaled, pct_cutoffs)
    for i in range(max_iters):
        one_mark = np.where((value_scaled >= cutoffs[0]))
        two_marks = np.where((value_scaled >= cutoffs[1]))
        methyl = np.zeros(num_beads, dtype=int)
        methyl[one_mark] = 1
        methyl[two_marks] = 2
        obs_frac_methyl = np.sum(methyl) / (2 * num_beads)
        err = obs_frac_methyl - fraction_methylated

        if -tolerance <= err <= tolerance:
            print("Convergence Successful!")
            return methyl.astype(int)
        
        cutoffs[0] += avg_value * err * (i / max_iters)
        cutoffs[1] += 2 * avg_value * err * (i / max_iters)

        if i == (max_iters-1):
            print("Failed to converge.")

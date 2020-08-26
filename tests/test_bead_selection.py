# Test bead_selection from left
# For all tests, use a selection window size of 10000 beads.

from chromo.util.bead_selection import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, anderson_ksamp


def plot_distribution_of_bead_selections(selections, discrete_bins, x_theoretical, y_theoretical, x_axis_label, file_name):
    """
    Plot distribution of expected/observed bead selections.

    Parameters
    ----------
    selections : np.array
        1D array of bead selection indices from sample
    discrete_bins : np.array
        1D array of bin edges between which selections are distributed
    x_theoretical : np.array
        1D array of all bead indices
    y_theoretical : np.array
        1D array of probabilities corresponding to indices in x_theoretical
    x_axis_label : str
        Label to put on x-axis of plot
    file_name : str
        Path from cwd to save file
    """
    plt.hist(selections, bins = discrete_bins, density = True)
    plt.plot(x_theoretical, y_theoretical)
    plt.xlabel(x_axis_label)
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close() 
    
    return


def batch_bead_selection(num_beads, select_from_left = True):
    """
    Exponentially select from 10000 beads during 10000 trials.
    
    Parameters
    ----------
    num_beads : int
        Number of beads in polymer
    select_from_left : boolean
        True (default) if exponentially selecting beads from LHS of polymer; False if selecting from RHS; None if from point
    Returns
    -------
    selections : np.array
        1D array of bead indices exponentially selected
    select_densities : np.array
        1D array of bead selection probabilities within each bin
    select_bin_edges : np.array
        1D array of bins for bead selection probabilities
    """

    window = num_beads - 1
    discrete_bins = np.arange(0, num_beads, 100)
    exclude_last_bead = True
    num_trials = 10000

    selections = np.zeros(num_trials)

    if select_from_left == True:        # Testing LHS rotation
        for i in range(num_trials): 
            selections[i] = select_bead_from_left(window, num_beads, exclude_last_bead)
    elif select_from_left == False:     # Testing RHS rotation
        for i in range(num_trials):
            selections[i] = select_bead_from_right(window, num_beads, exclude_last_bead)    
    elif select_from_left == None:      # Testing rotation from point
        for i in range(num_trials):
            selections[i] = select_bead_from_point(window, num_beads, round(num_beads/2))
        
    select_densities, select_bin_edges = np.histogram(selections, bins = discrete_bins, density = True)

    return selections, select_densities, select_bin_edges


def compare_with_expected_selections(num_beads, select_densities, select_bin_edges, select_from_left = True):
    """
    Compare randomly selected beads with an exponential distribution.

    Parameters
    ----------
    num_beads : int
        Number of beads in polymer
    select_densities : np.array
        1D array of bead selection probabilities within each bin
    select_bin_edges : np.array
        1D array of bins for bead selection probabilities
    select_from_left : boolean
        True (default) if exponentially selecting beads from LHS of polymer; False if selecting from RHS
    Returns
    -------
    x : np.array
        1D array of all bead indices
    y : np.array
        1D array of probabilities corresponding to indices in x_theoretical
    float
        AD-test statistic comparing observed and expected distributions   
    """
    x = np.arange(0, num_beads, 1)
    y = expon.pdf(x, 0, num_beads-1) / (expon.cdf(num_beads-1, 0, num_beads-1) - expon.cdf(0, 0, num_beads-1))

    if select_from_left == False:
        y = np.flip(y)
 
    y_discrete, _ = np.histogram(y, bins = select_bin_edges)
    
    return x, y, anderson_ksamp([select_densities, y_discrete])[0]    


def test_select_bead_from_left():
    """
    Test for exponentially distributed bead selection from LHS.
    """
    num_beads = 10000
    selections, select_densities, select_bin_edges = batch_bead_selection(num_beads)
    x, y, AD_stat = compare_with_expected_selections(num_beads, select_densities, select_bin_edges)
    # plot_distribution_of_bead_selections(selections, select_bin_edges, x, y, "Index Selection from Left", "tests/Exponential_Sampling_from_Left.PNG")
    assert AD_stat < 200    # Trial runs turn up an AD-test statistic close to 175. 
    assert y[0] > y[-1]     # For selection from left, first bead has higher probability than last bead


def test_select_bead_from_right():
    """
    Test for exponentially distributed bead selection from RHS.
    """
    num_beads = 10000
    selections, select_densities, select_bin_edges = batch_bead_selection(num_beads, select_from_left = False)
    x, y, AD_stat = compare_with_expected_selections(num_beads, select_densities, select_bin_edges, select_from_left = False)
    # plot_distribution_of_bead_selections(selections, select_bin_edges, x, y, "Index Selection from Right", "tests/Exponential_Sampling_from_Right.PNG")
    assert AD_stat < 200    # Trial runs turn up an AD-test statistic close to 175. 
    assert y[0] < y[-1]     # For selection from right, first bead has lower probability than last bead
    

def test_select_bead_from_point():
    """
    Test for exponentially distributed bead selection from point.
    """
    num_beads = 10000
    window = num_beads-1
    ind0 = round(num_beads/2)
    selections, select_densities, select_bin_edges = batch_bead_selection(num_beads, select_from_left = None)

    # Produce expected distribution
    x = np.arange(0, num_beads, 1)
    y = expon.pdf(abs(ind0-x), 0, window) / (2 * (expon.cdf(ind0, 0, window) - expon.cdf(0, 0, window)))
    y_discrete, _ = np.histogram(y, bins = select_bin_edges)
    AD_stat, _, _ = anderson_ksamp([select_densities, y_discrete])
    # plot_distribution_of_bead_selections(selections, select_bin_edges, x, y, "Index Selection from Center", "tests/Exponential_Sampling_from_Center.PNG")
    assert AD_stat < 200    # Trial runs turn up an AD-test statistic close to 175. 


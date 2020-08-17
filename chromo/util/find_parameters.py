"""Routines for calculating the potential forces on the beads."""
import numpy as np


def find_parameters(length_bead):
    """Determine the parameters for the elastic forces based on ssWLC with twist."""
    sim_type = "sswlc"
    lp_dna = 53     # Persistence length of DNA in nm
    length_dim = length_bead * 0.34 / lp_dna        # Non-dimensionalized length per bead
    param_values = np.loadtxt("chromo/util/dssWLCparams")    # Load the parameter table from file

    # Determine the parameter values using linear interpolation of the parameter table
    eps_bend = np.interp(length_dim, param_values[:, 0], param_values[:, 1]) / length_dim
    gamma = np.interp(length_dim, param_values[:, 0], param_values[:, 2]) * length_dim * lp_dna
    eps_par = np.interp(length_dim, param_values[:, 0], param_values[:, 3]) / (length_dim * lp_dna**2)
    eps_perp = np.interp(length_dim, param_values[:, 0], param_values[:, 4]) / (length_dim * lp_dna**2)
    eta = np.interp(length_dim, param_values[:, 0], param_values[:, 5]) / lp_dna

    return sim_type, eps_bend, eps_par, eps_perp, gamma, eta

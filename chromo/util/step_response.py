"""Response characterization by step changes in MC simulation parameters."""

import numpy as np

from chromo.mc.adapt import performance_to_file


def amp_bead_step_adapter(mc_adapter):
    """
    Step function applied to bead amplitude for response characterization.

    Use this function to characterize changes in move acceptance rate in
    response to step changes in simulation parameters. This allows us to
    model the response as a first order process and fit control scheme for
    move acceptance rates.

    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate

    Returns
    -------
    MCAdapter Object
        MC move parameters updated based on step function
    """
    mc_adapter, _, _, _, _ = performance_to_file(mc_adapter)
    
    if mc_adapter.performance_tracker.step_count == 5000:
        mc_adapter.amp_bead = 10
    elif mc_adapter.performance_tracker.step_count == 10000:
        mc_adapter.amp_bead = 15
    elif mc_adapter.performance_tracker.step_count == 15000:
        mc_adapter.amp_bead = 25
    elif mc_adapter.performance_tracker.step_count == 20000:
        mc_adapter.amp_bead = 50
    elif mc_adapter.performance_tracker.step_count == 25000:
        mc_adapter.amp_bead = 75
    elif mc_adapter.performance_tracker.step_count == 30000:
        mc_adapter.amp_bead = 90
    elif mc_adapter.performance_tracker.step_count == 45000:
        mc_adapter.amp_bead = 95
    
    return mc_adapter


def amp_move_step_adapter(mc_adapter):
    """
    Step function applied to move amplitude for response characterization.

    Use this function to characterize changes in move acceptance rate in
    response to step changes in simulation parameters. This allows us to
    model the response as a first order process and fit control scheme for
    move acceptance rates.

    Parameters
    ----------
    mc_adapter : MCAdapter Object
        MC move parameters for adaption based on move acceptance rate

    Returns
    -------
    MCAdapter Object
        MC move parameters updated based on step function
    """
    mc_adapter, _, _, _, _ = performance_to_file(mc_adapter)

    if mc_adapter.performance_tracker.step_count == 5000:
        mc_adapter.amp_move = 1.5 * np.pi
    elif mc_adapter.performance_tracker.step_count == 10000:
        mc_adapter.amp_move = 2 * np.pi
    elif mc_adapter.performance_tracker.step_count == 15000:
        mc_adapter.amp_move = 2.5 * np.pi
    elif mc_adapter.performance_tracker.step_count == 20000:
        mc_adapter.amp_move = 3 * np.pi
    elif mc_adapter.performance_tracker.step_count == 25000:
        mc_adapter.amp_move = 3.5 * np.pi
    elif mc_adapter.performance_tracker.step_count == 30000:
        mc_adapter.amp_move = 4 * np.pi
    elif mc_adapter.performance_tracker.step_count == 45000:
        mc_adapter.amp_move = 4.5 * np.pi

    return mc_adapter
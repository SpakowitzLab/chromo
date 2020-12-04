"""Test feedback adaptor for MC simulator parameters."""

import numpy as np

from chromo.mc.moves import MCAdapter
from chromo.mc.moves import slide
from chromo.mc.adapt import PerformanceTracker
from chromo.mc.adapt import feedback_adaption

def test_feedback_adaption():
    """Test feedback adaption to changes in acceptance rate."""
    
    # Generate an MC adapter instance
    mc_adapter = MCAdapter(slide)
    # Overwrite `amp_bead` and `amp_move`
    mc_adapter.amp_move = 100
    mc_adapter.amp_bead = 100
    # Overwrite acceptance rate & setpoints to force changes in `amp_bead`
    mc_adapter.thresholds = [0, 0.5, 1]
    mc_adapter.window_factor = [0.75, 0.95]
    mc_adapter.move_factor = [0.75, 0.95]
    mc_adapter.bead_amp_range = [0, 1000]
    mc_adapter.move_amp_range = [0, 1000]

    ## TEST DECREASE BEAD AMPLITUDE
    mc_adapter.performance_tracker.tracked_steps = [0] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert np.isclose(mc_adapter.amp_bead, 75)

    ## TEST INCREASE BEAD AMPLITUDE
    mc_adapter.amp_bead = 100
    mc_adapter.performance_tracker.tracked_steps = [0.9] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert(np.isclose(round(mc_adapter.amp_bead, 2), 126.58))

    ## TEST DECREASE IN BEAD AMPLITUDE BELOW LIMIT
    mc_adapter.bead_amp_range = [80, 110]
    mc_adapter.amp_bead = 100
    mc_adapter.performance_tracker.tracked_steps = [0] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert np.isclose(mc_adapter.amp_bead, 80)

    ## TEST INCREASE IN BEAD AMPLITUDE ABOVE LIMIT
    mc_adapter.amp_bead = 100
    mc_adapter.performance_tracker.tracked_steps = [0.9] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert(np.isclose(round(mc_adapter.amp_bead, 2), 110))

    ## TEST DECREASE IN MOVE AMPLITUDE
    mc_adapter.thresholds = [0.5, 0.5, 0.5]
    mc_adapter.performance_tracker.tracked_steps = [0] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert np.isclose(round(mc_adapter.amp_move, 2), 75)

    ## TEST INCREASE IN MOVE AMPLITUDE
    mc_adapter.amp_move = 100
    mc_adapter.performance_tracker.tracked_steps = [0.9] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert(np.isclose(round(mc_adapter.amp_move, 2), 126.58))

    ## TEST DECREASE IN MOVE AMPLITUDE BELOW LIMIT
    mc_adapter.move_amp_range = [80, 110]
    mc_adapter.amp_move = 100
    mc_adapter.performance_tracker.tracked_steps = [0] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert np.isclose(mc_adapter.amp_move, 80)

    ## TEST INCREASE IN MOVE AMPLITUDE ABOVE LIMIT
    mc_adapter.amp_move = 100
    mc_adapter.performance_tracker.tracked_steps = [0.9] * \
        mc_adapter.performance_tracker.num_steps_tracked
    feedback_adaption(mc_adapter)
    assert(np.isclose(round(mc_adapter.amp_move, 2), 110))


def test_performance_tracker():
    """ Test methods in performance tracker object."""

    n_track = 50
    n_start = 100

    performance_tracker = PerformanceTracker(
        num_steps_tracked=n_track, startup_steps=n_start)

    # Test startup
    for i in range(n_start + 1):
        performance_tracker.step_count += 1
        performance_tracker.add_step(accepted=True)
    assert np.sum(performance_tracker.tracked_steps) == n_track
    assert performance_tracker.startup == False

    # Test queue
    for i in range(n_track):
        performance_tracker.add_step(accepted=False)
    assert np.sum(performance_tracker.tracked_steps) == 0

    # Test `calc_acceptance_rate`
    for i in range(25):
        performance_tracker.add_step(accepted=True)
    assert performance_tracker.calc_acceptance_rate() == 0.5

    # Test redo startup
    performance_tracker.redo_startup()
    assert performance_tracker.startup == True
    assert performance_tracker.tracked_steps == [None] * n_track

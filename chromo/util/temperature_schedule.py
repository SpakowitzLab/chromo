import numpy as np


class Schedule:
    """Class representation of the simulated annealing schedule.

    Notes
    -----
    By wrapping the simulated annealing schedules in a class, the functions
    become compatible with `reproducibility.py`.
    """

    def __init__(self, fxn):
        self.function = fxn
        self.name = fxn.__name__

    def to_file(self, path):
        with open(path, 'w') as f:
            f.write(self.name)


def decreasing_stepwise(current_step, total_steps):
    step_size = (1 - 0.1) / total_steps
    value = max(0.1, 1 - step_size * current_step)

    return value
def logarithmic_decrease(current_step, total_steps):
    if current_step == 0:
    	return 1
    ratio = current_step/total_steps
    result = -1 * np.log(ratio/2)/5.5 -0.1
    if result > 1:
        result = 1
    if result < 0:
        result = 0
    return result

def linear_decrease(current_step, total_steps):
    slope = (0.1 - 1) / total_steps
    value = 1 + slope * current_step
    return value
def no_schedule():
    return 1


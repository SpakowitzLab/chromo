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

def sin_decrease(current_step, total_steps):
    slope = (-1.3) / total_steps
    linear_value = 1 + slope * current_step
    result = np.sin(current_step/total_steps *100) 
    result = result *.1 + linear_value *.6 +.32
    if result > 1:
        result = 1
    if result < 0.1:
        result = 0.1
    return result * 10


def decreasing_stepwise(current_step, total_steps):
    raw_fraction = 1 - current_step/(total_steps + 150)
    fraction = raw_fraction * 10
    step_fraction = np.floor(fraction)/10
    return step_fraction

def logarithmic_decrease(current_step, total_steps):
    if current_step == 0:
    	return 1
    ratio = current_step/total_steps
    result = -1 * np.log(ratio/2)/5.5 -0.1
    if result > 1:
        result = 1
    if result < 0:
        result = 0
    return result * 10

def linear_decrease(current_step, total_steps):
    slope = (0.1 - 1) / total_steps
    value = 1 + slope * current_step
    return value
def no_schedule():
    return 1


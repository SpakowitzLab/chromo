"""Chemical Potential Schedules to test for simulated annealing.

In all functions, `i` denotes the current snapshot index and `N` denotes the
total number of snapshots that will be collected. Both `i` and `N` should be
integers. `i` should be less than `N`.
"""

import numpy as np


class Schedule:
    def __init__(self, fxn):
        self.function = fxn
        self.name = fxn.__name__

    def to_file(self, path):
        with open(path, 'w') as f:
            f.write(self.name)


def linear_1(i, N):
    """Linear schedule between -1 and 1
    """
    return 2 * (i / N) - 1


def linear_2(i, N):
    """Linear schedule between -1 and 1, but with delayed start.
    """
    delay = np.floor(N/4)
    if i <= delay:
        return -1
    else:
        return 2 * (float(i - delay) / float(N - delay)) - 1


def linear_3(i, N):
    """Linear schedule between -1 and 1, but with delayed start.
    """
    delay = np.floor(N/6)
    if i <= delay:
        return -1
    else:
        return 2 * (float(i - delay) / float(N - delay)) - 1


def linear_4(i, N):
    """Linear schedule between -1 and 1, but with delayed start.
    """
    delay = np.floor(N/8)
    if i <= delay:
        return -1
    else:
        return 2 * (float(i - delay) / float(N - delay)) - 1


def tanh_1(i, N):
    """Hyperbolic tangent schedule between -1 and 1.
    """
    return np.tanh(5 * float(i) / N - 3)


def tanh_2(i, N):
    """Gradual hyperbolic tangent schedule between -0.7 and 1.
    """
    return np.tanh(3 * float(i) / N - 1)


def tanh_3(i, N):
    """Gradual hyperbolic tangent schedule between -0.7 and 1 w/ delayed start.
    """
    delay = np.floor(N/6)
    if i <= delay:
        return np.tanh(-1)
    else:
        return np.tanh(3 * float(i-delay) / float(N-delay) - 1)


def tanh_4(i, N):
    """Gradual hyperbolic tangent schedule between -0.7 and 1 w/ delayed start.
    """
    delay = np.floor(N/8)
    if i <= delay:
        return np.tanh(-1)
    else:
        return np.tanh(3 * (i-delay) / (N-delay) - 1)


def log_1(i, N):
    """Logarithmically increasing mu schedule between -1 and 1.
    """
    return np.log(i / (N / 6.25) + 1) - 1


def log_2(i, N):
    """Logarithmically increasing mu schedule to 1.
    """
    return np.log(i / (N / 4) + 0.4) - 0.5


def log_3(i, N):
    """Logarithmically increasing mu schedule to 1.
    """
    return np.log(i/(N/2.5) + 0.5) - 0.098


def sawtooth_1(i, N):
    """Decaying sawtooth temperature schedule.
    """
    num_tooth = 20
    num_steps = np.floor(i/num_tooth)
    if (N - i) > 2:
        return np.log(i / (N/10) + 0.5) - 0.04 - 0.15 * num_steps
    else:
        return 1


def linear_2_for_negative_cp(i, N):
    """Linear schedule between 4 and 1, with delayed start.
    """
    delay = np.floor(N/4)
    if i <= delay:
        return 4.
    else:
        return -3. * (float(i - delay) / float(N - delay)) + 4


def linear_step_for_negative_cp(i, N):
    """Linear Schedule between 4 and 1, with delayed start and early end.
    """
    start = np.floor(N/5)
    end = np.floor(N/5)
    if i <= start:
        return 4.
    elif i >= (N - end):
        return 1.
    else:
        return -3. * (float(i - start) / float(N - (start + end))) + 4

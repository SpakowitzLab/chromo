
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

def increasing_stepwise(current_step, total_steps):
    step_size = (1 - 0.1) / total_steps
    value = max(0.1, 1 - step_size * current_step)

    return value
def logarithmic_increase(current_step, total_steps):
    #value = 5
    value =  (current_step/total_steps) * 100
    return value

def linear_increase(current_step, total_steps):
    value =  (current_step/total_steps) * 100
    return value
def no_schedule():
    return 100
#dynamic structure factor
#density correlation function at differnet lengthscales
#fouirer transform of density density correlation function
#how much largest lengthscale of polymer is changing
#correleation function of radius of gyration
#how density fluctluation within polymer correlate in time
#plot rmsd for different twist values
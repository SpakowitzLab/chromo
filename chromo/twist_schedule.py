
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

def step_wise_increase(current_step, total_steps):
    num_blocks = 10
    max_height = 100
    step_height = max_height/num_blocks # each step has a height of 10
    step_length = total_steps /num_blocks # so 20 is the length
    division = current_step/step_length # so if we are at snapshot 105 we get 5.11
    ceiling = np.ceil(division) # we are currently on the 6th step
    result = step_height * ceiling
    return result

    return value
def logarithmic_increase(current_step, total_steps):
    if current_step == 0:
      return 0
    ratio = current_step/total_steps
    result =  np.log(ratio/2)/5.5 *100 + 113
    if result > 100:
      result = 100
    if result < 0:
      result = 0
    return result

def linear_increase(current_step, total_steps):
    max_value = 100
    slope = max_value/total_steps
    value = slope * current_step
    return value
def increasing_sawtooth(current_step, total_steps):
    num_blocks = 10
    max_height = 100
    step_height = max_height/num_blocks # each step has a height of 10
    step_length = total_steps /num_blocks # so 20 is the length
    division = current_step/step_length # so if we are at snapshot 105 we get 5.11
    ceiling = np.ceil(division) # we are currently on the 6th step
    result = step_height * ceiling # this is the height we are at in the 6th block

    num_sections = num_blocks * 2 #if we are at 105
    section_division = np.floor(current_step/(total_steps/num_sections)) # 10.5 is the section from 20 total sections
    if section_division%2 == 0:
      # we are in section 6
      #result = result + (current_step%num_sections) - section_division 
      result = result + (current_step%num_sections)*2 - section_division -20
    else:
      result = result - (current_step%num_sections)*2 - section_division + 20
    return result



def no_schedule():
    return 100

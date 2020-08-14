# Test bead_selection from left
# For all tests, use a selection window size of 10000 beads.

from chromo.util.bead_selection import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def test_select_bead_from_left():
	num_beads = 10000
	all_beads = np.arange(1, num_beads + 1, 1)
	exclude_last_bead = True
	num_trials = 10000
	selections = []

	for i in range(0, num_trials):
		selections.append(
			select_bead_from_left(num_beads, all_beads, exclude_last_bead)
			)

	selections = np.array(selections)

	plt.hist(selections, bins = 25)
	plt.xlabel("Index Selection from Left")
	plt.ylabel("Frequency")
	plt.savefig("tests/outputs/Exponential_Sampling_from_Left.PNG")
	plt.close()
	assert True
	
	return


def test_select_bead_from_right():
	num_beads = 10000
	all_beads = np.arange(1, num_beads + 1, 1)
	exclude_last_bead = True
	num_trials = 10000
	selections = []

	for i in range(0, num_trials):
		selections.append(
			select_bead_from_right(num_beads, all_beads, exclude_last_bead)
			)

	selections = np.array(selections)

	plt.hist(selections, bins = 25)
	plt.xlabel("Index Selection from Right")
	plt.ylabel("Frequency")
	plt.savefig("tests/outputs/Exponential_Sampling_from_Right.PNG")
	plt.close()
	assert True

	return


def test_select_bead_from_point():
	num_beads = 10000
	all_beads = np.arange(1, num_beads + 1, 1)
	exclude_last_bead = True
	ind0 = round(num_beads / 2)
	num_trials = 10000
	selections = []

	for i in range(0, num_trials):
		selections.append(
			select_bead_from_point(num_beads, all_beads, ind0)
			)

	selections = np.array(selections)

	plt.hist(selections, bins = 25)
	plt.xlabel("Index Selection from Central Point: " + str(ind0))
	plt.ylabel("Frequency")
	plt.savefig("tests/outputs/Exponential_Sampling_from_Center.PNG")
	plt.close()	
	assert True

	return


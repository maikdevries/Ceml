import numpy as np


def construct (R):
	"""
	Construct a new R-dimensional uniform vector as EG expert weights.
	"""
	return np.ones(R, dtype = np.float64) / R


def select_expert (m, R):
	"""
	Randomly select an expert based on the EG expert weights vector (m).
	"""

	# Randomly select an expert according to the expert advice weights vector (m)
	random_expert = np.random.default_rng().choice(R, p = m)

	# Generate R-dimensional vector with a single randomly selected expert set to True
	r = np.zeros(R, dtype = bool)
	r[random_expert] = True

	return r


def calc_utility (r, U):
	"""
	Calculate utility of selected expert (r) based on combined utility vector (U).
	"""
	return np.sum(r * U)


def update (U, m, learning_rate):
	"""
	Perform the exponentiated gradient step of the EG algorithm - returned vector is inside feasible solution set.
	"""
	z = m * np.exp(learning_rate * U)

	# Normalise the updated expert advice weights vector (z) to ensure it sums up to 1
	return z / np.sum(z)

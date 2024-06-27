import numpy as np


def construct (T, K):
	"""
	Construct a new T-by-K continuous matrix as EG expert probability weights.
	"""
	M = np.zeros((T, K), dtype = np.float64)
	M[0] = 1 / K

	return M


def select_expert (m, K):
	"""
	Randomly select a caching expert based on the EG expert probability weights vector (m).
	"""

	# Randomly select a caching expert according to the expert probability weights vector (m)
	random_expert = np.random.default_rng().choice(K, p = m)

	# Generate a K-dimensional boolean vector with a single randomly selected expert set to True
	k = np.zeros(K, dtype = bool)
	k[random_expert] = True

	return k


def calc_learning_rate (L, T, K):
	"""
	Calculate the learning rate of the exponentiated gradient algorithm.
	"""
	return np.sqrt((2 * np.log(K)) / (np.power(L, 2) * T))


def calc_utility (k, u):
	"""
	Calculate the utility of the selected expert (k) based on the combined utility vector (u).
	"""
	return np.sum(k * u)


def update (m, u, delta):
	"""
	Perform the exponentiated gradient step of the EG algorithm - returned vector is inside feasible solution set.
	"""
	z = m * np.exp(delta * u)

	# Normalise the updated expert probability weights vector (z) to ensure it sums up to 1
	return z / np.sum(z)

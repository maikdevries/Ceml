import numpy as np


def construct (X, W, N, C):
	"""
	Construct a new N-dimensional continuous vector as BSCH cache configuration.
	"""

	# Calculate the achieved utility of each file for request matrix (X) and file weights (W)
	utilities = np.sum(W * X, axis = 0)

	# Retrieve the indices that would partition the array into the C highest utility-scoring files
	indices = np.argpartition(utilities, -C)[-C:]

	# Generate an N-dimensional vector with the highest utility-scoring files set to 1.0 (fully cached)
	Y = np.zeros(N, dtype = np.float64)
	Y[indices] = 1.0

	return Y


def calc_utility (X, Y, W):
	"""
	Calculate the utility of request matrix (X) for cache configuration (Y) and static file weights (W).
	"""
	return np.sum(W * X * Y, axis = 1)

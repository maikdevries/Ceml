import numpy as np


def construct (X, W, N, C):
	"""
	Construct a new N-dimensional vector as BSH cache configuration.
	"""

	# Calculate the frequency of each file in the request matrix X
	frequencies = np.sum(X, axis = 0)

	# Calculate the utility gained in regard to the file request frequencies
	utilities = W * frequencies

	# Retrieve the indices that would partition the array into the C highest utility-scoring files
	indices = np.argpartition(utilities, -C)[-C:]

	# Generate N-dimensional vector with the highest utility-scoring files set to 1.0 (fully cached)
	Y = np.zeros(N, dtype = np.float64)
	Y[indices] = 1.0

	return Y


def calc_utility (X, Y, W):
	"""
	Calculate utility of request matrix (X) for cache configuration (Y) and static file weights (W).
	"""

	# Generate a T-by-N matrix where each row contains the file request frequency up to and including the current timeslot
	frequencies = np.cumsum(X, axis = 0)

	return np.sum(W * frequencies * Y, axis = 1)

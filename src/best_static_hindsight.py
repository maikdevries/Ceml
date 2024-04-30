import numpy as np


def calc_utility (X, W, C):
	"""
	Calculate the utility of the best static cache configuration in hindsight.
	"""

	# Calculate the frequency for each file request in X
	frequencies = np.sum(X, axis = 0)

	# Calculate the utility gained in regard to file request frequencies
	utility = W * frequencies

	# Retrieve the indices that would partition the array into the C highest utility-scoring files
	indices = np.argpartition(utility, -C)[-C:]

	# Sum up the utility of the C highest scoring files
	return np.sum(utility[indices])

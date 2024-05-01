import numpy as np


def calc_utility (X, W, C):
	"""
	Given a N-dimensional request frequencies vector X, calculate the utility of the best static cache configuration in hindsight.
	"""

	# Calculate the utility gained in regard to file request frequencies
	utility = W * X

	# Retrieve the indices that would partition the array into the C highest utility-scoring files
	indices = np.argpartition(utility, -C)[-C:]

	# Sum up the utility of the C highest scoring files
	return np.sum(utility[indices])

import numpy as np

import online_gradient_ascent as OGA


def calc_utility_OGA (X, Y, W, T, N, C):
	"""
	Given a T-by-N request matrix X, calculate the utility of the Online Gradient Ascent algorithm.
	"""
	OGA_utility = []

	for x in X:

		# Calculate the (possibly dynamic) learning rate for current request x
		diam = OGA.calc_diam(N, C)
		L = OGA.calc_L(x, W)
		learning_rate = OGA.calc_learning_rate(diam, L, T)

		# Calculate the utility of the current OGA cache configuration
		OGA_utility.append(OGA.calc_utility(x, Y, W))

		# TODO: avoid (costly) update of cache configuration on last request in X
		# Update OGA cache configuration based on gradient of request
		z = OGA.online_gradient_ascent(x, Y, W, learning_rate)
		Y = OGA.project(z, N, C)

	return np.sum(OGA_utility)


def calc_utility_hindsight (X, W, C):
	"""
	Given a T-by-N request matrix X, calculate the utility of the best caching configuration in hindsight.
	"""

	# Calculate the frequency for each file request in X
	frequencies = np.sum(X, axis = 0)

	# Calculate the utility gained in regard to file request frequencies
	utility = W * frequencies

	# Retrieve the indices that would partition the array into the C highest utility-scoring files
	indices = np.argpartition(utility, -C)[-C:]

	# Sum up the utility of the C highest scoring files
	return np.sum(utility[indices])

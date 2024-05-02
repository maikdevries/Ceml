import numpy as np

import best_static_hindsight as BSH
import online_gradient_ascent as OGA
import least_recently_used as LRU


def calc_utility_BSH (X, W, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the best static caching configuration in hindsight.
	"""

	# Generate a T-by-N matrix where each row contains the file request frequency up to and including the current timeslot
	frequencies = np.cumsum(X, axis = 0)

	return np.apply_along_axis(BSH.calc_utility, axis = 1, arr = frequencies, W = W, C = C)


def calc_utility_OGA (X, W, T, N, C, R):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the online gradient ascent algorithm.
	"""
	cache = OGA.construct(N)
	utility = []

	for x in X:

		# Calculate dynamic learning rate for current request x if not provided
		if not R:
			diam = OGA.calc_diam(N, C)
			L = OGA.calc_L(x, W)
			R = OGA.calc_learning_rate(diam, L, T)

		# Calculate the utility of the current OGA cache configuration
		utility.append(OGA.calc_utility(x, cache, W))

		# TODO: avoid (costly) update of cache configuration on last request in X
		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.update(x, cache, W, R)
		cache = OGA.project(z, N, C)

	return np.asarray(utility, dtype = np.float64).cumsum()


def calc_utility_LRU (X, W, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the least recently used caching policy.
	"""
	cache = LRU.construct(C)
	utility = []

	for x in X:

		# Update LRU cache configuration and calculate utility based on whether current request x was a cache hit or miss
		if LRU.update(x, cache):
			utility.append(LRU.calc_utility(x, W))
		else:
			utility.append(0)

	return np.asarray(utility, dtype = np.float64).cumsum()

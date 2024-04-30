import numpy as np

import best_static_hindsight as BSH
import online_gradient_ascent as OGA
import least_recently_used as LRU


def calc_utility_BSH (X, W, T, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the best static caching configuration in hindsight.
	"""
	utility = []

	for t in range(1, T + 1):

		# Calculate the utility sum of all requests in X up to and including timeslot t
		utility.append(BSH.calc_utility(X[:t], W, C))

	return np.asarray(utility, dtype = np.float64)


def calc_utility_OGA (X, W, T, N, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the online gradient ascent algorithm.
	"""
	cache = OGA.construct(N)
	utility = []

	for x in X:

		# Calculate the (possibly dynamic) learning rate for current request x
		diam = OGA.calc_diam(N, C)
		L = OGA.calc_L(x, W)
		learning_rate = OGA.calc_learning_rate(diam, L, T)

		# Calculate the utility of the current OGA cache configuration
		utility.append(OGA.calc_utility(x, cache, W))

		# TODO: avoid (costly) update of cache configuration on last request in X
		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.update(x, cache, W, learning_rate)
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

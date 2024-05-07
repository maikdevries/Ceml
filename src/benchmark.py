import time
import numpy as np

import best_static_hindsight as BSH
import online_gradient_ascent as OGA
import least_recently_used as LRU


def calc_utility_BSH (X, W, N, C, start_time = time.perf_counter()):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the best caching configuration in hindsight.
	"""
	cache = BSH.construct(X, W, N, C)

	return (
		BSH.calc_utility(X, cache, W),
		cache,
		time.perf_counter() - start_time,
	)


def calc_utility_OGA (X, W, T, N, C, R, start_time = time.perf_counter()):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the online gradient ascent algorithm.
	"""
	cache = OGA.construct(N)
	utility = []

	# Loop over all requests in X, except last request to avoid unnecessary yet costly update of cache configuration
	for x in X[:-1]:

		# Calculate the utility of the current OGA cache configuration
		utility.append(OGA.calc_utility(x, cache, W))

		# Calculate dynamic learning rate for current request x if not provided
		if R is None:
			diam = OGA.calc_diam(N, C)
			L = OGA.calc_L(x, W)
			R = OGA.calc_learning_rate(diam, L, T)

		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.update(x, cache, W, R)
		cache = OGA.project(z, N, C)

	# Calculate the utility of the last request in X, without updating the cache configuration
	utility.append(OGA.calc_utility(X[-1], cache, W))

	return (
		np.asarray(utility, dtype = np.float64).cumsum(),
		cache,
		time.perf_counter() - start_time,
	)


def calc_utility_LRU (X, W, N, C, start_time = time.perf_counter()):
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

	return (
		np.asarray(utility, dtype = np.float64).cumsum(),
		LRU.to_vector(cache, N),
		time.perf_counter() - start_time,
	)

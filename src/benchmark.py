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
	state = []
	utility = []

	# Loop over all requests in X, except last request to avoid unnecessary yet costly update of cache configuration
	for x in X[:-1]:

		# Store the current OGA cache configuration and calculate its utility
		state.append(cache.copy())
		utility.append(OGA.calc_utility(x, cache, W))

		# Calculate dynamic learning rate for current request x if not provided
		if R is None:
			diam = OGA.calc_diam(N, C)
			L = OGA.calc_L(x, W)
			R = OGA.calc_learning_rate(diam, L, T)

		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.update(x, cache, W, R)
		cache = OGA.project(z, N, C)

	# Store the current OGA cache configuration and calculate the utility of the last request in X, without updating the cache configuration
	state.append(cache.copy())
	utility.append(OGA.calc_utility(X[-1], cache, W))

	return (
		np.asarray(utility, dtype = np.float64).cumsum(),
		np.asarray(state, dtype = np.float64),
		time.perf_counter() - start_time,
	)


def calc_utility_LRU (X, W, N, C, start_time = time.perf_counter()):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the least recently used caching policy.
	"""
	cache = LRU.construct(C)
	state = []
	utility = []

	for x in X:

		# Store the current LRU cache configuration
		state.append(LRU.to_vector(cache, N))

		# Update LRU cache configuration and calculate utility based on whether current request x was a cache hit or miss
		if LRU.update(x, cache):
			utility.append(LRU.calc_utility(x, W))
		else:
			utility.append(0)

	return (
		np.asarray(utility, dtype = np.float64).cumsum(),
		np.asarray(state, dtype = np.float64),
		time.perf_counter() - start_time,
	)

import numpy as np
import time

from src import (
    best_static_configuration_hindsight as BSCH,
    least_recently_used as LRU,
    online_gradient_ascent as OGA,
    exponentiated_gradient as EG,
)


def calc_utility_BSCH (X, W, N, C, start_time = time.perf_counter()):
	"""
	Given a T-by-N request matrix (X), accumulate the utility of the best static configuration in hindsight.
	"""
	Y = BSCH.construct(X, W, N, C)

	return (
		BSCH.calc_utility(X, Y, W),
		Y,
		time.perf_counter() - start_time,
	)


def calc_utility_OGA (X, W, T, N, C, eta, BSCH_cache, start_time = time.perf_counter()):
	"""
	Given a T-by-N request matrix (X), accumulate the utility of the online gradient ascent caching policy.
	"""
	Y = OGA.construct(N)
	cache_distance = []
	utility = []

	for t in range(T):

		# Store the current OGA cache configuration's (Y) Euclidean distance to BSCH and calculate its utility
		cache_distance.append(np.linalg.norm(BSCH_cache - Y))
		utility.append(OGA.calc_utility(X[t], Y, W))

		# Calculate dynamic learning rate for current request (X[t]) if not provided
		if eta is None:
			diam = OGA.calc_diam(N, C)
			L = OGA.calc_L(X[t], W)
			eta = OGA.calc_learning_rate(diam, L, T)

		# Update OGA cache configuration (Y) based on the gradient of the current request (X[t]) and project back onto feasible solution set
		z = OGA.update(X[t], Y, W, eta)
		Y = OGA.project(z, N, C)

	return (
		np.asarray(utility, dtype = np.float64),
		np.asarray(cache_distance, dtype = np.float64),
		time.perf_counter() - start_time,
	)


def calc_utility_LRU (X, W, T, N, C, BSCH_cache, start_time = time.perf_counter()):
	"""
	Given a T-by-N request matrix (X), accumulate the utility of the least recently used caching policy.
	"""
	Y = LRU.construct(C)
	cache_distance = []
	utility = []

	for t in range(T):

		# Store the current LRU cache configuration's (Y) Euclidean distance to BSCH
		cache_distance.append(np.linalg.norm(BSCH_cache - LRU.to_vector(Y, N)))

		# Update LRU cache configuration (Y) and calculate its utility based on whether the current request (X[t]) was a cache hit or miss
		if LRU.update(X[t], Y):
			utility.append(LRU.calc_utility(X[t], W))
		else:
			utility.append(0)

	return (
		np.asarray(utility, dtype = np.float64),
		np.asarray(cache_distance, dtype = np.float64),
		time.perf_counter() - start_time,
	)


def calc_utility_EG (U, T, K, start_time = time.perf_counter()):
	"""
	Given a T-by-K utility matrix (U), accumulate the utility of the exponentiated gradient meta learner.
	"""
	M = EG.construct(T + 1, K)
	utility = []

	# Calculate the learning rate (delta) based on the maximum utility value in U
	delta = EG.calc_learning_rate(np.max(U), T, K)

	for t in range(T):

		# Randomly pick a caching expert based on the current expert probability weights (M[t])
		k = EG.select_expert(M[t], K)

		# Accumulate the achieved utility of selected caching expert k at current time slot t
		utility.append(EG.calc_utility(k, U[t]))

		# Calculate expert probability weights M[t + 1] based on the gradient of U[t]
		M[t + 1] = EG.update(M[t], U[t], delta)

	return (
		np.asarray(utility, dtype = np.float64),
		M,
		time.perf_counter() - start_time,
	)

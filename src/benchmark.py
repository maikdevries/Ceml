import numpy as np

import best_static_hindsight as BSH
import online_gradient_ascent as OGA
import least_recently_used as LRU


def compare_utility_OGA_LRU (X, Y, W, T, N, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of OGA and LRU caching policies.
	"""
	OGA_utility = []
	LRU_utility = []

	LRU_cache = LRU.LRU(C)

	for x in X:

		# Calculate the (possibly dynamic) learning rate for current request x
		diam = OGA.calc_diam(N, C)
		L = OGA.calc_L(x, W)
		learning_rate = OGA.calc_learning_rate(diam, L, T)

		# Calculate the utility of the current OGA cache configuration
		OGA_utility.append(OGA.calc_utility(x, Y, W))

		# TODO: avoid (costly) update of cache configuration on last request in X
		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.online_gradient_ascent(x, Y, W, learning_rate)
		Y = OGA.project(z, N, C)

		# Update LRU cache configuration and calculate utility based on whether current request x was a cache hit or miss
		if LRU_cache.update(x):
			LRU_utility.append(np.sum(W * x))
		else:
			LRU_utility.append(0)

	return (OGA_utility, LRU_utility)


def compare_utility_OGA_hindsight (X, Y, W, T, N, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of OGA and the best static caching configuration in hindsight.
	"""
	OGA_utility = []
	hindsight_utility = []

	for t, x in enumerate(X):

		# Calculate the (possibly dynamic) learning rate for current request x
		diam = OGA.calc_diam(N, C)
		L = OGA.calc_L(x, W)
		learning_rate = OGA.calc_learning_rate(diam, L, T)

		# Calculate the utility of the current OGA cache configuration
		OGA_utility.append(OGA.calc_utility(x, Y, W))

		# TODO: avoid (costly) update of cache configuration on last request in X
		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.online_gradient_ascent(x, Y, W, learning_rate)
		Y = OGA.project(z, N, C)

		# Calculate the utility of the best caching configuration in hindsight
		hindsight_utility.append(BSH.calc_utility(X[:(t + 1)], W, C))

	return (OGA_utility, hindsight_utility)


def calc_utility_BSH (X, W, T, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the best static caching configuration in hindsight.
	"""
	BSH_utility = []

	for t in range(T):

		# Calculate the utility sum of all requests in X up to and including timeslot t
		BSH_utility.append(BSH.calc_utility(X[:(t + 1)], W, C))

	return BSH_utility


def calc_utility_OGA (X, Y, W, T, N, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the online gradient ascent algorithm.
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
		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.online_gradient_ascent(x, Y, W, learning_rate)
		Y = OGA.project(z, N, C)

	return OGA_utility

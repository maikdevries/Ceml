import numpy as np

import online_gradient_ascent as OGA
import least_recently_used as LRU
import generate


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
	Given a T-by-N request matrix X, accumulate the utility of OGA and the best caching configuration in hindsight.
	"""
	OGA_utility = []
	hindsight_utility = []

	Y_hindsight = generate.optimal_Y_hindsight(X, W, N, C)

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

		# Calculate the utility of the best caching configuration in hindsight
		hindsight_utility.append(OGA.calc_utility(x, Y_hindsight, W))

	return (OGA_utility, hindsight_utility)


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
		# Update OGA cache configuration based on gradient of request and project back onto feasible solution set (constraints)
		z = OGA.online_gradient_ascent(x, Y, W, learning_rate)
		Y = OGA.project(z, N, C)

	return np.sum(OGA_utility)


def calc_utility_hindsight (X, W, C):
	"""
	Given a T-by-N request matrix X, calculate the utility of the best caching configuration in hindsight.
	"""
	utility = generate.calc_request_utility(X, W)

	# Retrieve the indices that would partition the array into the C highest utility-scoring files
	indices = np.argpartition(utility, -C)[-C:]

	# Sum up the utility of the C highest scoring files
	return np.sum(utility[indices])

import best_static_hindsight as BSH
import online_gradient_ascent as OGA
import least_recently_used as LRU


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


def calc_utility_LRU (X, W, C):
	"""
	Given a T-by-N request matrix X, accumulate the utility of the least recently used caching policy.
	"""
	LRU_cache = LRU.construct(C)
	LRU_utility = []

	for x in X:

		# Update LRU cache configuration and calculate utility based on whether current request x was a cache hit or miss
		if LRU.update(x, LRU_cache):
			LRU_utility.append(LRU.calc_utility(x, W))
		else:
			LRU_utility.append(0)

	return LRU_utility

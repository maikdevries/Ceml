import numpy as np
from collections import deque


def construct (C):
	"""
	Construct a new LRU cache configuration with a given maximum size (C).
	"""
	return deque(maxlen = C)


def update (x, y: deque):
	"""
	Update the LRU cache configuration (y) based on the given request vector (x).
	"""

	# Determine index of requested file in request vector (x)
	index = np.argmax(x)

	# Append to end of cache if requested file is not in cache and return False (cache miss)
	if index not in y:
		y.append(index)
		return False

	# Remove requested file from cache and append to end of cache if requested file is in cache and return True (cache hit)
	y.remove(index)
	y.append(index)

	return True


def calc_utility (x, w):
	"""
	Calculate utility of request instance (x) and static file weights (w) in case of LRU cache hit.
	"""
	return np.sum(w * x)

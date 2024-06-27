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

	# Determine the index of the requested file in request vector (x)
	index = np.argmax(x)

	# Append to the end of the cache if the requested file is not in the cache and return False (cache miss)
	if index not in y:
		y.append(index)
		return False

	# Remove the requested file from the cache and append to end if the requested file is in the cache and return True (cache hit)
	y.remove(index)
	y.append(index)

	return True


def calc_utility (x, w):
	"""
	Calculate the utility of the request instance (x) and file weights (w) in case of a LRU cache hit.
	"""
	return np.sum(w * x)

from collections import deque
from numpy import argmax

class LRU:
	"""
	Class representing a Least Recently Used (LRU) cache configuration.
	"""

	def __init__ (self, C):
		"""
		Construct a LRU cache configuration with a limited capacity C.
		"""
		self.cache = deque(maxlen = C)


	def update (self, x):
		"""
		Update the LRU cache configuration based on the given request vector (x).
		"""

		# Determine index of requested file in request vector (x)
		index = argmax(x)

		# Append to end of cache if requested file is not in cache and return False (cache miss)
		if index not in self.cache:
			self.cache.append(index)
			return False

		# Remove requested file from cache and append to end of cache if requested file is in cache and return True (cache hit)
		self.cache.remove(index)
		self.cache.append(index)

		return True

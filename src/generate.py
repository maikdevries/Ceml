import numpy as np


def X_random (T, N):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random element set to True.
	"""

	# Generate T random indices between 0 and N (exclusive)
	random_indices = np.random.default_rng().integers(0, N, size = T)

	# Generate a T-by-N zero matrix and set a random element of each vector to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), random_indices] = True

	return X


def X_random_bounded (T, N, B):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random element in range [0 .. B] set to True.
	"""
	assert B > 0 and B <= N, 'The upper bound B must be within range [1 .. N]'

 	# Generate T random indices between 0 and B (exclusive)
	random_indices = np.random.default_rng().integers(0, B, size = T)

 	# Generate a T-by-N zero matrix and set a random element of each vector to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), random_indices] = True

	return X


def X_zipfian (T, N, zeta = 0.8):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random element set to True according to a Zipfian
	distribution with parameter zeta.
	"""
	assert zeta >= 0, 'The Zipfian distribution parameter zeta must be non-negative'

	# Approximate probabilities following Zipf's law for N elements and parameter zeta
	probabilities = np.power(np.arange(1, N + 1), -zeta)
	probabilities /= np.sum(probabilities)

	# Generate T random indices between 0 and N (exclusive) according to Zipf's law
	random_indices = np.random.default_rng().choice(N, size = T, p = probabilities)

	# Generate a T-by-N zero matrix and set a random element of each vector to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), random_indices] = True

	return X


def X_round_robin (T, N, B):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single element set to True in a round-robin fashion.
	"""
	assert B > 0 and B <= N, 'The round-robin upper bound B must be within range [1 .. N]'

	# Generate T indices between 0 and B (exclusive) in a round-robin fashion
	random_indices = np.arange(T) % B

	# Generate a T-by-N zero matrix and set a random element of each vector to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), random_indices] = True

	return X


def Y_zero (N):
	"""
	Generate an N-dimensional zero vector.
	"""
	return np.zeros(N, dtype = np.float64)


def Y_random (N, C):
	"""
	Generate an N-dimensional vector with each element in range [0 .. 1) and all its elements summing up to at most C.
	"""

	# Generate an N-dimensional vector with its elements randomly set in range [0 .. 1)
	Y = np.random.default_rng().random(N)

	# Scale Y such that all its elements sum up to at most C
	Y *= (C / np.sum(Y))

	return Y


def W_uniform (N):
	"""
	Generate an N-dimensional boolean vector with each element set to True.
	"""
	return np.ones(N, dtype = bool)


def W_random (N):
	"""
	Generate an N-dimensional continuous vector with each element in range [0 .. 1).
	"""
	return np.random.default_rng().random(N)

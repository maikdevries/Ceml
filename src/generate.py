import numpy as np


# TODO: refactor random X generators into single function definition with desired distribution parameter
def random_X (T, N):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry set to True.
	"""

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	# Generate T random indices between 0 and N (exclusive)
	random_indices = np.random.randint(0, N, size = T)

	# Generate T-by-N zero matrix and set a random element of each vector to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), random_indices] = True

	return X


def random_X_zipfian (T, N, alpha = 0.8):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry set to True according to a Zipfian
	distribution.
	"""
	assert alpha >= 0, 'The Zipfian distribution parameter alpha must be non-negative'

	# Approximate probabilities following Zipf's law for N elements and parameter alpha
	probabilities = np.power(np.arange(1, N + 1), -alpha)
	probabilities /= np.sum(probabilities)

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	# Generate T random indices between 0 and N (exclusive) according to Zipf's law
	random_indices = np.random.choice(N, size = T, p = probabilities)

	# Generate T-by-N zero matrix and set a random element of each vector to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), random_indices] = True

	return X


def adversarial_X_round_robin (T, N, B):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry set to True in a round-robin fashion.
	"""
	assert B > 0 and B <= N, 'The round-robin upper bound B must be within range [1 .. N]'

	# Generate T indices between 0 and B (exclusive) in a round-robin fashion
	round_robin_indices = np.arange(T) % B

	# Generate T-by-N zero matrix and set element of each vector at the corresponding round-robin index to True
	X = np.zeros((T, N), dtype = bool)
	X[np.arange(T), round_robin_indices] = True

	return X


def zero_Y (N):
	"""
	Generate an N-dimensional zero vector.
	"""
	return np.zeros(N, dtype = np.float64)


def random_Y (N, C):
	"""
	Generate an N-dimensional vector with each element in range [0 .. 1] and all elements sum up to at most C.
	"""

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	# Generate N-dimensional vector with its elements randomly set in range [0 .. 1]
	Y = np.random.rand(N)

	# Scale Y such that all elements sum up to at most C
	Y *= (C / np.sum(Y))

	return Y


def uniform_weights (N):
	"""
	Generate an N-dimensional boolean vector with each element set to True.
	"""
	return np.ones(N, dtype = bool)


def random_weights (N):
	"""
	Generate an N-dimensional floating-point vector with each element in range [0 .. 1].
	"""

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	return np.random.rand(N)

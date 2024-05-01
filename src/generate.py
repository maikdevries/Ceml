import numpy as np


def zero_X (T, N):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional zero vectors.
	"""
	return np.zeros((T, N), dtype = bool)


# TODO: refactor random X generators into single function definition with desired distribution parameter
def random_X (T, N):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry set to True.
	"""

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	# Generate T random indices between 0 and N (exclusive)
	random_indices = np.random.randint(0, N, size = T)

	# Generate T-by-N zero matrix and set a random element of each vector to True
	X = zero_X(T, N)
	X[np.arange(T), random_indices] = True

	return X


def random_X_zipf (T, N, alpha = 1.25):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry set to True according to Zipf's law.
	"""

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	# Generate T random indices between 0 and N (exclusive) according to Zipf's law
	random_indices = (np.random.zipf(alpha, size = T) - 1) % N

	# Generate T-by-N zero matrix and set a random element of each vector to True
	X = zero_X(T, N)
	X[np.arange(T), random_indices] = True

	return X


def adversarial_X_round_robin (T, N):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry set to True in a round-robin fashion.
	"""

	# Generate T indices between 0 and N (exclusive) in a round-robin fashion
	round_robin_indices = np.arange(T) % N

	# Generate T-by-N zero matrix and set element of each vector at the corresponding round-robin index to True
	X = zero_X(T, N)
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

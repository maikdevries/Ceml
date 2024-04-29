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
	random_indices = np.random.zipf(alpha, size = T) % N

	# Generate T-by-N zero matrix and set a random element of each vector to True
	X = zero_X(T, N)
	X[np.arange(T), random_indices] = True

	return X


def random_bounded_X (T, N, B):
	"""
	Generate a T-by-N boolean matrix which contains T N-dimensional vectors with a single random entry in range [0 .. B] set to True.
	"""

	# TODO: replace np.random function call with corresponding Generator instance (recommended implementation)
	# Generate T random indices between 0 and B (exclusive)
	random_indices = np.random.randint(0, B, size = T)

	# Generate T-by-N zero matrix and set random element of each vector to True
	X = zero_X(T, N)
	X[np.arange(T), random_indices] = True

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


def calc_request_utility (X, W):
	"""
	Given a T-by-N request matrix X, calculate the total utility per request based on frequency.
	"""

	# Calculate the frequency for each file request in X
	frequencies = np.sum(X, axis = 0)

	# Calculate the utility gained in regard to file request frequencies
	return W * frequencies


def optimal_Y_hindsight (X, W, N, C):
	"""
	Given a T-by-N request matrix X, generate the best static caching configuration in hindsight.
	"""
	utility = calc_request_utility(X, W)

	# Retrieve the indices of C files which result in the highest utility given their respective request frequencies
	indices = np.argsort(utility)[-C:]

	# Generate N-dimensional vector with the highest utility-earning files set to 1.0 (fully cached)
	Y = zero_Y(N)
	Y[indices] = 1.0

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

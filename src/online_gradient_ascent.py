import numpy as np
import cvxpy as cp
import math


# Define global variables for speed-up by reusing them in repeated instances of the minimisation problem
feasible_vector = None
unfeasible_vector = None
minimisation_problem = None


def construct (N):
	"""
	Construct a new N-dimensional zero vector as OGA cache configuration.
	"""
	return np.zeros(N, dtype = np.float64)


def calc_diam (N, C):
	"""
	Calculate the diameter of the feasible solution set (Y).
	"""
	boundary_condition = N / 2

	if 0 < C and C <= boundary_condition:
		return math.sqrt(2 * C)
	elif boundary_condition < C and C <= N:
		return math.sqrt(2 * (N - C))


def calc_L (x, w):
	"""
	Calculate the upper bound of the gradient of the utility function.
	"""
	return np.max(np.sum(w * x))


def calc_learning_rate (diam, L, T):
	"""
	Calculate the learning rate of the online gradient ascent algorithm.
	"""
	return diam / (L * math.sqrt(T))


def calc_utility (x, y, w):
	"""
	Calculate the utility of the request instance (x) for cache configuration instance (y) and static file weights (w).
	"""
	return np.sum(w * x * y)


def calc_gradient (x, w):
	"""
	Calculate the gradient of the utility function with respect to the cache configuration instance (y).
	"""
	return w * x


def update (x, y, w, eta):
	"""
	Perform the gradient ascent step of the OGA algorithm - returned vector might be outside feasible solution set.
	"""
	return y + (eta * calc_gradient(x, w))


def define_minimisation_problem (N, C):
	"""
	Instantiate global variables for the minimisation problem such that they can be reused in repeated instances.
	"""
	global feasible_vector, unfeasible_vector, minimisation_problem

	# Define an N-dimensional non-negative feasible target vector which is to be closest to the unfeasible vector
	feasible_vector = cp.Variable(N, nonneg = True)

	# Define a mutable parameter to be used in the immutable minimisation problem
	unfeasible_vector = cp.Parameter(N, nonneg = True)

	# Define the minimisation problem of squared Euclidean distance and constraints of the feasible solution set:
	# A vector's elements should be in range [0 .. 1] and the vector's elements should sum up to at most C.
	minimisation_problem = cp.Problem(
		cp.Minimize(cp.sum_squares(unfeasible_vector - feasible_vector)),
		[
			feasible_vector <= 1,
			cp.sum(feasible_vector) <= C,
		],
	)


# TODO: replace CVXPY abstraction with direct use of solver (?)
def project (z, N, C):
	"""
	Project the (unfeasible) vector (z) onto the feasible solution set through minimisation of squared Euclidean distance.
	"""
	global feasible_vector, unfeasible_vector, minimisation_problem

	# Define the global variables if not yet instantiated
	if feasible_vector is None or unfeasible_vector is None or minimisation_problem is None:
		define_minimisation_problem(N, C)

	assert feasible_vector is not None and unfeasible_vector is not None and minimisation_problem is not None

	# Update the unfeasible vector parameter with the current (unfeasible) vector (z)
	unfeasible_vector.value = z

	# TODO: test different canonicalisation backends (CPP, SCIPY, NUMPY)
	# TODO: test different solvers for speed and accuracy (CLARABEL, OSQP, ECOS, SCS)
	minimisation_problem.solve(solver = cp.CLARABEL, enforce_dpp = True)

	return feasible_vector.value

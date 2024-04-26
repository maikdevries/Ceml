import numpy as np
import cvxpy as cp
import math


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
	Calculate the (possibly dynamic) learning rate.
	"""
	return diam / (L * math.sqrt(T))


def calc_utility (x, y, w):
	"""
	Calculate utility of request instance (x) for cache configuration instance (y) and (static) request weights (w).
	"""
	return np.sum(w * x * y)


def calc_gradient (x, w):
	"""
	Calculate gradient of utility function with respect to cache configuration instance (y).
	"""
	return w * x


def online_gradient_ascent (x, y, w, learning_rate):
	"""
	Perform the gradient ascent step of the OGA algorithm - returned vector might be outside feasible solution set.
	"""
	return y + (learning_rate * calc_gradient(x, w))


# TODO: replace CVXPY abstraction with direct use of solver (?)
def project (z, N, C):
	"""
	Project (unfeasible) vector (z) onto feasible solution set through minimisation of squared Euclidean distance.
	"""

	# Define N-dimensional feasible target vector which is to be closest to (unfeasible) vector z
	x = cp.Variable(N)

	# Define minimisation problem of squared Euclidean distance and constraints of the feasible solution set:
	# A vector's elements should be in range [0 .. 1], and additionally the vector's elements should sum up to at most C.
	problem = cp.Problem(
		cp.Minimize(cp.sum_squares(z - x)),
		[
			x >= 0,
			x <= 1,
			cp.sum(x) <= C,
		],
	)

	# TODO: test different solvers for speed and accuracy (CLARABEL, OSQP, ECOS, SCS)
	problem.solve(solver = cp.CLARABEL)

	return x.value

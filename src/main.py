import numpy as np
import matplotlib.pyplot as plt

import generate
import benchmark


# The number of time slots, and consequently the number of requests or iterations (horizon)
T = 1000

# The size of the system's library of files and cache, where C should be (significantly) smaller than N
N = 1000
C = 100

# T-by-N matrix: at each timeslot t a single entry is set to True (file request)
X = generate.random_X_zipf(T, N)

# N-dimensional vector: at each timeslot t all entries will sum up to at most C (cache configuration)
Y = generate.random_Y(N, C)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)


# TODO: assert C is within range [0 .. N]
if __name__ == '__main__':

	# Calculate sum of utility for various caching policies
	# OGA_utility = benchmark.calc_utility_OGA(X, Y, W, T, N, C)
	# hindsight_utility = benchmark.calc_utility_hindsight(X, W, C)

	# Retrieve lists of utility progression over time for various caching policies
	(OGA_utility, hindsight_utility) = benchmark.compare_utility_OGA_hindsight(X, Y, W, T, N, C)

	sum_OGA = np.sum(OGA_utility)
	sum_hindsight = np.sum(hindsight_utility)

	print('Utility accumulated by OGA policy:', sum_OGA)
	print('Utility accumulated by best static cache configuration in hindsight:', sum_hindsight)
	print('Regret achieved by OGA policy:', (sum_hindsight - sum_OGA))

	# Create moving average filter to smooth out strong variations in achieved utility (noise)
	moving_average_filter = np.ones(50) / 50

	(fig, ax) = plt.subplots()
	fig.suptitle(f'Average request utility over time [N = {N}, C = {C}]')
	fig.supxlabel('Timeslot')
	fig.supylabel('Utility')

	ax.plot(np.convolve(OGA_utility, moving_average_filter, mode = 'valid'), label = 'OGA')
	ax.plot(np.convolve(hindsight_utility, moving_average_filter, mode = 'valid'), label = 'Hindsight')

	plt.legend()
	plt.show()

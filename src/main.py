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

	# Retrieve lists of utility progression over time for various caching policies
	BSH_utility = benchmark.calc_utility_BSH(X, W, T, C)
	OGA_utility = benchmark.calc_utility_OGA(X, Y, W, T, N, C)

	# TODO: return numpy array instead of list
	BSH_utility = np.asarray(BSH_utility)
	OGA_utility = np.asarray(OGA_utility)

	print('Utility accumulated by BSH policy:', BSH_utility[-1])
	print('Utility accumulated by OGA policy:', np.sum(OGA_utility))
	print('Regret achieved by OGA policy:', (BSH_utility[-1] - np.sum(OGA_utility)))

	# Create moving average filter to smooth out strong variations in achieved utility (noise)
	moving_average_filter = np.ones(50) / 50
	time_slots = np.arange(1, T + 1)

	(fig, ax) = plt.subplots()
	fig.suptitle(f'Average request utility over time [N = {N}, C = {C}]')
	fig.supxlabel('Time')
	fig.supylabel('Average utility')

	ax.plot(BSH_utility / time_slots, label = 'Hindsight')
	ax.plot(np.convolve(OGA_utility, moving_average_filter, mode = 'valid'), label = 'OGA')

	plt.legend()
	plt.show()

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

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)


# TODO: assert C is within range [0 .. N]
if __name__ == '__main__':

	# Retrieve lists of utility progression over time for various caching policies
	BSH_utility = benchmark.calc_utility_BSH(X, W, C)
	OGA_utility = benchmark.calc_utility_OGA(X, W, T, N, C)

	print(f'Utility accumulated by BSH policy: {BSH_utility[-1]}')
	print(f'Utility accumulated by OGA policy: {OGA_utility[-1]}')
	print(f'Regret achieved by OGA policy: {(BSH_utility[-1] - OGA_utility[-1])}')

	(fig, ax) = plt.subplots()
	fig.suptitle(f'Average request utility over time [N = {N}, C = {C}]')
	fig.supxlabel('Time')
	fig.supylabel('Average utility')

	# Generate list of time slots to be used in deriving average utility per time slot
	time_slots = np.arange(1, T + 1)

	ax.plot((BSH_utility / time_slots), label = 'BSH')
	ax.plot((OGA_utility / time_slots), label = 'OGA')

	plt.legend()
	plt.show()

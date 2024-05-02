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
X = generate.random_X_zipfian(T, N, alpha = 0.8)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)

# The learning rate for the online gradient ascent algorithm
R = None


# TODO: assert C is within range [0 .. N]
if __name__ == '__main__':

	# Retrieve lists of utility progression over time for various caching policies
	BSH_utility = benchmark.calc_utility_BSH(X, W, C)
	OGA_utility = benchmark.calc_utility_OGA(X, W, T, N, C, R)

	print(f'Utility accumulated by BSH policy: {BSH_utility[-1]}')
	print(f'Utility accumulated by OGA policy: {OGA_utility[-1]}')
	print(f'Regret achieved by OGA policy: {(BSH_utility[-1] - OGA_utility[-1])}')

	(fig, (dist, util)) = plt.subplots(2, 1)
	fig.suptitle(f'Average request utility over time [N = {N}, C = {C}, R = {R}]')

	dist.set_ylabel('Requested file')

	util.set_ylabel('Average utility')
	util.set_xlabel('Time')

	# Plot the requested file at each time slot (request distribution)
	dist.plot(np.argmax(X, axis = 1), '.', alpha = 0.5)

	# Generate list of time slots to be used in deriving average utility per time slot
	time_slots = np.arange(1, T + 1)

	util.plot((BSH_utility / time_slots), label = 'BSH')
	util.plot((OGA_utility / time_slots), label = 'OGA')

	plt.legend()
	plt.show()

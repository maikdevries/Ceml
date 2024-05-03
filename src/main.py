import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

import generate
import benchmark


# The number of time slots, and consequently the number of requests or iterations (horizon)
T = 50000

# The size of the system's library of files and cache, where C should be (significantly) smaller than N
N = 5000
C = 500

# T-by-N matrix: at each timeslot t a single entry is set to True (file request)
X = generate.random_X_zipfian(T, N, alpha = 0.8)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)

# Learning rates to be used in online gradient ascent algorithm (computed dynamically for None entries)
R = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]


# TODO: assert C is within range [0 .. N]
if __name__ == '__main__':

	# Calculate in parallel the utility progression over time for various caching policies
	with concurrent.futures.ProcessPoolExecutor() as executor:
		BSH_future = executor.submit(benchmark.calc_utility_BSH, X, W, C)
		LRU_future = executor.submit(benchmark.calc_utility_LRU, X, W, C)
		OGA_futures = [executor.submit(benchmark.calc_utility_OGA, X, W, T, N, C, r) for r in R]

		BSH_utility = BSH_future.result()
		LRU_utility = LRU_future.result()
		OGA_utilities = [future.result() for future in OGA_futures]

	print(f'Utility accumulated by BSH policy: {BSH_utility[-1]}')
	print(f'Utility accumulated by LRU policy: {LRU_utility[-1]}')

	for i, r in enumerate(R):
		print(f'Utility accumulated by OGA [{r}] policy: {OGA_utilities[i][-1]}')
		print(f'Regret achieved by OGA [{r}] vs BSH: {(BSH_utility[-1] - OGA_utilities[i][-1])}')
		print(f'Regret achieved by OGA [{r}] vs LRU: {(LRU_utility[-1] - OGA_utilities[i][-1])}')

	(fig, (dist, util)) = plt.subplots(2, 1)
	fig.suptitle(f'Average request utility over time [N = {N}, C = {C}]')

	dist.set_ylabel('File requests')

	util.set_ylabel('Average utility')
	util.set_xlabel('Time')

	# Plot total number of requests per file over horizon T (request distribution)
	dist.plot(np.sum(X, axis = 0))

	# Generate list of time slots to be used in deriving average utility per time slot
	time_slots = np.arange(1, T + 1)

	util.plot((BSH_utility / time_slots), label = 'BSH')
	util.plot((LRU_utility / time_slots), label = 'LRU')

	for i, r in enumerate(R):
		util.plot((OGA_utilities[i] / time_slots), label = f'OGA [{r}]')

	plt.legend()
	plt.show()

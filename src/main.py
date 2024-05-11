import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

from parameters import T, N, C, X, W, R
import benchmark


if __name__ == '__main__':

	# Calculate in parallel the utility progression over time for various caching policies
	with concurrent.futures.ProcessPoolExecutor() as executor:
		BSH_future = executor.submit(benchmark.calc_utility_BSH, X, W, N, C)
		LRU_future = executor.submit(benchmark.calc_utility_LRU, X, W, N, C)
		OGA_futures = [executor.submit(benchmark.calc_utility_OGA, X, W, T, N, C, r) for r in R]

		BSH_utility, BSH_cache, BSH_time = BSH_future.result()
		LRU_utility, LRU_caches, LRU_time = LRU_future.result()
		OGA_utilities, OGA_caches, OGA_times = zip(*[future.result() for future in OGA_futures])

	print(f'[{BSH_time:.2f}s] Utility accumulated by BSH policy: {BSH_utility[-1]:.2f}')
	print(f'[{LRU_time:.2f}s] Utility accumulated by LRU policy: {LRU_utility[-1]:.2f}')

	for i, r in enumerate(R):
		print(f'[{OGA_times[i]:.2f}s] Utility accumulated by OGA [{r}] policy: {OGA_utilities[i][-1]:.2f}')
		print(f'Regret achieved by OGA [{r}] vs BSH: {(BSH_utility[-1] - OGA_utilities[i][-1]):.2f}')
		print(f'Regret achieved by OGA [{r}] vs LRU: {(LRU_utility[-1] - OGA_utilities[i][-1]):.2f}')

	fig, (dist, hist, util) = plt.subplots(3, 1)
	fig.suptitle(f'Average request utility over time [T = {T}, N = {N}, C = {C}]')

	dist.set_ylabel('File requests')
	hist.set_ylabel('Distance to BSH cache')

	util.set_ylabel('Average utility')
	util.set_xlabel('Time slot')

	# Generate list of time slots to be used in deriving average utility per time slot
	time_slots = np.arange(1, T + 1)

	# Plot total number of requests per file over horizon T (request distribution)
	dist.plot(np.sum(X, axis = 0))

	# Plot the Euclidean distance between BSH cache configuration and other caching policies
	hist.plot(np.linalg.norm(BSH_cache - LRU_caches, axis = 1), label = 'LRU')

	util.plot((BSH_utility / time_slots), label = 'BSH')
	util.plot((LRU_utility / time_slots), label = 'LRU')

	for i, r in enumerate(R):
		hist.plot(np.linalg.norm(BSH_cache - OGA_caches[i], axis = 1), label = f'OGA [{r}]')
		util.plot((OGA_utilities[i] / time_slots), label = f'OGA [{r}]')

	plt.legend()
	plt.show()

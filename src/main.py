import concurrent.futures

from parameters import T, N, C, X, W, R
import benchmark
import output


if __name__ == '__main__':

	# Calculate in parallel the utility progression over time for various caching policies
	with concurrent.futures.ProcessPoolExecutor() as executor:
		BSH_future = executor.submit(benchmark.calc_utility_BSH, X, W, N, C)
		LRU_future = executor.submit(benchmark.calc_utility_LRU, X, W, N, C)
		OGA_futures = [executor.submit(benchmark.calc_utility_OGA, X, W, T, N, C, r) for r in R]

		BSH_utility, BSH_cache, BSH_time = BSH_future.result()
		LRU_utility, LRU_caches, LRU_time = LRU_future.result()
		OGA_utilities, OGA_caches, OGA_times = zip(*[future.result() for future in OGA_futures])

	# Print the running time of each caching policy to the console
	print(f'[{BSH_time:.2f}s] BSH cache policy')
	print(f'[{LRU_time:.2f}s] LRU cache policy')

	for i, r in enumerate(R):
		print(f'[{OGA_times[i]:.2f}s] OGA [{r}] cache policy')

	# Save the generated request matrix (X) to disk
	output.save_request_matrix(X, 'request_matrix')

	# Save the utility progression and cache configuration state(s) of each caching policy to disk
	output.save_results(BSH_utility, BSH_cache, 'BSH')
	output.save_results(LRU_utility, LRU_caches, 'LRU')

	for i, r in enumerate(R):
		output.save_results(OGA_utilities[i], OGA_caches[i], f'OGA_[{r}]')

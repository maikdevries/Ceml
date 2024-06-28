import concurrent.futures

from parameters import T, N, C, X, W, K
from src import benchmark, output


if __name__ == '__main__':

	# Calculate the utility progression for various caching policies in parallel
	with concurrent.futures.ProcessPoolExecutor() as executor:
		BSCH_future = executor.submit(benchmark.calc_utility_BSCH, X, W, N, C)
		LRU_future = executor.submit(benchmark.calc_utility_LRU, X, W, T, C)
		OGA_futures = [executor.submit(benchmark.calc_utility_OGA, X, W, T, N, C, k) for k in K]

		BSCH_utility, BSCH_time = BSCH_future.result()
		LRU_utility, LRU_time = LRU_future.result()
		OGA_utilities, OGA_times = zip(*[future.result() for future in OGA_futures])

	# Print the running time of each caching policy to the console
	print(f'[{BSCH_time:.2f}s] BSCH cache policy')
	print(f'[{LRU_time:.2f}s] LRU cache policy')

	for i, k in enumerate(K):
		print(f'[{OGA_times[i]:.2f}s] OGA [{k}] cache policy')

	# Save the generated request matrix (X) to disk
	output.save_request_matrix(X, 'request_matrix')

	# Save the generated file weights matrix (W) to disk
	output.save_file_weights(W, 'file_weights')

	# Save the utility progression of each caching policy to disk
	output.save_utility(BSCH_utility, 'BSCH')
	output.save_utility(LRU_utility, 'LRU')

	for i, k in enumerate(K):
		output.save_utility(OGA_utilities[i], f'OGA_[{k}]')

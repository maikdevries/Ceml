import concurrent.futures
import numpy as np

from parameters import R, L
from src import benchmark, output


if __name__ == '__main__':

	# Load the utility progression of each caching policy expert from disk
	OGA_utilities, _ = zip(*[output.load_results(f'OGA_[{r}]') for r in R])
	OGA_utilities = np.asarray(OGA_utilities).T

	# Calculate in parallel the utility progression over time for various meta-learners
	with concurrent.futures.ProcessPoolExecutor() as executor:
		EG_futures = [executor.submit(benchmark.calc_utility_EG, OGA_utilities, len(R), l) for l in L]

		EG_utilities, EG_weights, EG_times = zip(*[future.result() for future in EG_futures])

	# For each meta-learner print the running time to the console, and save the utility progression and weights to disk
	for i, l in enumerate(L):
		print(f'[{EG_times[i]:.2f}s] EG [{l}] meta learner')
		output.save_results(EG_utilities[i], EG_weights[i], f'EG_[{l}]')

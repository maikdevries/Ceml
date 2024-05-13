import numpy as np

from parameters import R, L
from src import benchmark, output


if __name__ == '__main__':

	# Load the utility progression of each caching policy expert from disk
	OGA_utilities, _ = zip(*[output.load_results(f'OGA_[{r}]') for r in R])

	# Calculate the utility progression over time for the meta learner
	EG_utility, EG_weights, EG_time = benchmark.calc_utility_EG(np.asarray(OGA_utilities).T, len(R), L)

	# Print the running time of the meta learner to the console
	print(f'[{EG_time:.2f}s] EG [{L}] meta learner')

	# Save the utility progression and weights states of the meta learner to disk
	output.save_results(EG_utility, EG_weights, f'EG_[{L}]')

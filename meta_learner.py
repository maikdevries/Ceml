import numpy as np

from parameters import T, K
from src import benchmark, output


if __name__ == '__main__':

	# Load the utility progression of each caching expert from disk
	OGA_utilities = zip(*[output.load_utility(f'OGA_[{k}]') for k in K])
	OGA_utilities = np.asarray(OGA_utilities).T

	# Calculate the utility progression for the meta-learner
	EG_utility, EG_weights, EG_time = benchmark.calc_utility_EG(OGA_utilities, T, len(K))

	# Print the running time of the meta-learner to the console
	print(f'[{EG_time:.2f}s] EG meta learner')

	# Save the utility progression of the meta-learner to disk
	output.save_utility(EG_utility, 'EG')

	# Save the expert weights progression of the meta-learner to disk
	output.save_weights(EG_weights, 'expert_weights')

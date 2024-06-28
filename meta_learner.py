import numpy as np

from parameters import T, K
from src import benchmark, output


if __name__ == '__main__':

	# Load the utility progression of each caching expert from disk
	expert_utilities = np.asarray([output.load(f'./data/policies/OGA_[{k:.2f}]') for k in K])

	# Calculate the utility progression for the meta-learner
	utility, expert_weights, running_time = benchmark.calc_utility_EG(expert_utilities.T, T, len(K))

	# Print the running time of the meta-learner to the console
	print(f'[{running_time:.2f}s] EG meta learner')

	# Save the expert weights progression of the meta-learner to disk
	output.save(expert_weights, './data/meta_learner/weights')

	# Save the utility progression of the meta-learner to disk
	output.save(utility, './data/meta_learner/utility')

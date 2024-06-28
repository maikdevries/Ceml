import numpy as np

from parameters import T, N, C, K
from src import output


if __name__ == '__main__':

	# Load the generated request matrix (X) from disk
	X = output.load('./data/inputs/request_matrix')

	# Load and accumulate the utility of each caching policy and the meta-learner expert weights from disk
	BSCH_utility = output.load('./data/policies/BSCH').cumsum()
	LRU_utility = output.load('./data/policies/LRU').cumsum()
	expert_utilities = np.asarray([output.load(f'./data/policies/OGA_[{k:.2f}]').cumsum() for k in K])
	meta_learner_utility = output.load('./data/meta_learner/utility').cumsum()
	meta_learner_weights = output.load('./data/meta_learner/weights')

	print(f'Utility accumulated by BSCH policy: {BSCH_utility[-1]:.2f}')
	print(f'Utility accumulated by LRU policy: {LRU_utility[-1]:.2f}')

	for i, k in enumerate(K):
		print(f'Utility accumulated by OGA [{k:.2f}] policy: {expert_utilities[i][-1]:.2f}')
		print(f'Regret achieved by OGA [{k:.2f}] vs BSCH: {(BSCH_utility[-1] - expert_utilities[i][-1]):.2f}')
		print(f'Regret achieved by OGA [{k:.2f}] vs LRU: {(LRU_utility[-1] - expert_utilities[i][-1]):.2f}')

	print(f'Utility accumulated by EG meta-learner: {meta_learner_utility[-1]:.2f}')

	# Plot the request distribution (X) over the horizon (T)
	output.plot_request_distribution(X, T)

	# Plot the utility progression of each caching expert
	output.plot_expert_utilities(expert_utilities, T, N, C, K)

	# Plot the caching expert weights progression of the meta-learner
	output.plot_meta_learner_weights(meta_learner_weights, N, C, K)

	# Plot the utility progression of the meta-learner
	output.plot_meta_learner_utility(meta_learner_utility, T, N, C)

import numpy as np

from parameters import T, N, C, K
from src import output


if __name__ == '__main__':

	# Load the generated request matrix (X) from disk
	X = output.load_request_matrix('request_matrix')

	# Load the utility progression and cache distances of each caching expert from disk
	BSCH_utility = output.load_utility('BSCH')
	LRU_utility = output.load_utility('LRU')
	OGA_utilities = [output.load_utility(f'OGA_[{k:.2f}]') for k in K]
	EG_utility = output.load_utility('EG')
	EG_weights = output.load_weights('expert_weights')

	# Sum the achieved utility over time to obtain the accumulated utility of each caching expert
	BSCH_utility = BSCH_utility.cumsum()
	LRU_utility = LRU_utility.cumsum()
	OGA_utilities = np.asarray([u.cumsum() for u in OGA_utilities])
	EG_utility = EG_utility.cumsum()

	print(f'Utility accumulated by BSCH policy: {BSCH_utility[-1]:.2f}')
	print(f'Utility accumulated by LRU policy: {LRU_utility[-1]:.2f}')

	for i, k in enumerate(K):
		print(f'Utility accumulated by OGA [{k:.2f}] policy: {OGA_utilities[i][-1]:.2f}')
		print(f'Regret achieved by OGA [{k:.2f}] vs BSCH: {(BSCH_utility[-1] - OGA_utilities[i][-1]):.2f}')
		print(f'Regret achieved by OGA [{k:.2f}] vs LRU: {(LRU_utility[-1] - OGA_utilities[i][-1]):.2f}')

	print(f'Utility accumulated by EG meta-learner: {EG_utility[-1]:.2f}')

	# Plot the request distribution (X) over the horizon (T)
	output.plot_request_distribution(X, T)

	# Plot the utility progression of each caching expert
	output.plot_expert_utilities(OGA_utilities, T, N, C, K)

	# Plot the caching expert weights progression of the meta-learner
	output.plot_meta_learner_weights(EG_weights, N, C, K)

	# Plot the utility progression of the meta-learner
	output.plot_meta_learner_utility(EG_utility, T, N, C)

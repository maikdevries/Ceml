import numpy as np

from parameters import T, N, C, R, L
from src import output


if __name__ == '__main__':

	# Load the generated request matrix from disk
	X = output.load_request_matrix('request_matrix')

	# Load the utility progression and cache configuration state(s) of each caching policy expert from disk
	BSH_utility, BSH_cache = output.load_results('BSH')
	LRU_utility, LRU_caches = output.load_results('LRU')
	OGA_utilities, OGA_caches = zip(*[output.load_results(f'OGA_[{r}]') for r in R])
	EG_utilities, EG_weights = zip(*[output.load_results(f'EG_[{l}]') for l in L])

	# Sum the achieved utility over time to obtain the accumulated utility
	BSH_utility = BSH_utility.cumsum()
	LRU_utility = LRU_utility.cumsum()
	OGA_utilities = np.asarray([u.cumsum() for u in OGA_utilities])
	EG_utilities = np.asarray([u.cumsum() for u in EG_utilities])

	print(f'Utility accumulated by BSH policy: {BSH_utility[-1]:.2f}')
	print(f'Utility accumulated by LRU policy: {LRU_utility[-1]:.2f}')

	for i, r in enumerate(R):
		print(f'Utility accumulated by OGA [{r}] policy: {OGA_utilities[i][-1]:.2f}')
		print(f'Regret achieved by OGA [{r}] vs BSH: {(BSH_utility[-1] - OGA_utilities[i][-1]):.2f}')
		print(f'Regret achieved by OGA [{r}] vs LRU: {(LRU_utility[-1] - OGA_utilities[i][-1]):.2f}')

	for i, l in enumerate(L):
		print(f'Utility accumulated by EG [{l}] policy: {EG_utilities[i][-1]:.2f}')
		print(f'Regret achieved by EG [{l}] vs BSH: {(BSH_utility[-1] - EG_utilities[i][-1]):.2f}')
		print(f'Regret achieved by EG [{l}] vs LRU: {(LRU_utility[-1] - EG_utilities[i][-1]):.2f}')

	# Plot the request distribution over the horizon T
	output.plot_request_distribution(X, T)

	# Plot the Euclidean distance of each caching policy expert to the BSH cache configuration
	output.plot_expert_distances([np.linalg.norm(BSH_cache - cache, axis = 1) for cache in OGA_caches], N, C, R)

	# Plot the utility progression of each caching policy expert
	output.plot_expert_utilities(OGA_utilities, T, N, C, R)

	# Plot the caching policy expert weights of the meta-learner
	output.plot_meta_learner_weights(EG_weights, R, L)

	# Plot the utility progression of the meta-learner
	output.plot_meta_learner_utilities(EG_utilities, T, N, C, L)

import numpy as np
import matplotlib.pyplot as plt

from parameters import T, N, C, R


def save_request_matrix (X, file_name):
	"""
	Save the request matrix (X) to disk.
	"""
	with open(f'./results/{file_name}.npy', 'wb') as f:
		np.save(f, X)


def load_request_matrix (file_name):
	"""
	Load the request matrix from disk.
	"""
	with open(f'./results/{file_name}.npy', 'rb') as f:
		X = np.load(f)

	return X


def save_results (utility, state, file_name):
	"""
	Save the utility progression and cache configuration state(s) of a caching policy to disk.
	"""
	with open(f'./results/utility/{file_name}.npy', 'wb') as f:
		np.save(f, utility)

	with open(f'./results/state/{file_name}.npy', 'wb') as f:
		np.save(f, state)


def load_results (file_name):
	"""
	Load the utility progression and cache configuration state(s) of a caching policy from disk.
	"""
	with open(f'./results/utility/{file_name}.npy', 'rb') as f:
		utility = np.load(f)

	with open(f'./results/state/{file_name}.npy', 'rb') as f:
		state = np.load(f)

	return (utility, state)


if __name__ == '__main__':

	# Load the generated request matrix (X) from disk
	X = load_request_matrix('request_matrix')

	# Load the utility progression and cache configuration state(s) of each caching policy from disk
	BSH_utility, BSH_cache = load_results('BSH')
	LRU_utility, LRU_caches = load_results('LRU')
	OGA_utilities, OGA_caches = zip(*[load_results(f'OGA_[{r}]') for r in R])

	# Sum the achieved utility over time to obtain the accumulated utility
	BSH_utility = BSH_utility.cumsum()
	LRU_utility = LRU_utility.cumsum()
	OGA_utilities = np.asarray([u.cumsum() for u in OGA_utilities])

	print(f'Utility accumulated by BSH policy: {BSH_utility[-1]:.2f}')
	print(f'Utility accumulated by LRU policy: {LRU_utility[-1]:.2f}')

	for i, r in enumerate(R):
		print(f'Utility accumulated by OGA [{r}] policy: {OGA_utilities[i][-1]:.2f}')
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

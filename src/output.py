import numpy as np


def save_request_matrix (X, file_name):
	"""
	Save the request matrix (X) to disk.
	"""
	with open(f'./results/{file_name}.npy', 'wb') as f:
		np.save(f, X)


def save_results (utility, state, file_name):
	"""
	Save the utility progression and cache configuration state(s) of a caching policy to disk.
	"""
	with open(f'./results/utility/{file_name}.npy', 'wb') as f:
		np.save(f, utility)

	with open(f'./results/state/{file_name}.npy', 'wb') as f:
		np.save(f, state)

import numpy as np
import matplotlib.pyplot as plt


def save_request_matrix (X, file_name):
	"""
	Save the request matrix (X) to disk.
	"""
	with open(f'./data/{file_name}.npy', 'wb') as f:
		np.save(f, X)


def load_request_matrix (file_name):
	"""
	Load the request matrix from disk.
	"""
	with open(f'./data/{file_name}.npy', 'rb') as f:
		X = np.load(f)

	return X


def save_results (utility, state, file_name):
	"""
	Save the utility progression and cache configuration state(s) of a caching policy to disk.
	"""
	with open(f'./data/utility/{file_name}.npy', 'wb') as f:
		np.save(f, utility)

	with open(f'./data/state/{file_name}.npy', 'wb') as f:
		np.save(f, state)


def load_results (file_name):
	"""
	Load the utility progression and cache configuration state(s) of a caching policy from disk.
	"""
	with open(f'./data/utility/{file_name}.npy', 'rb') as f:
		utility = np.load(f)

	with open(f'./data/state/{file_name}.npy', 'rb') as f:
		state = np.load(f)

	return (utility, state)


def plot_request_distribution (X, T):
	"""
	Plot the request distribution (X) over the horizon (T).
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'File request distribution\n[T = {T}]')

	ax.set_ylabel('Number of requests')
	ax.set_xlabel('File')

	# Plot total number of requests per file over horizon T (request distribution)
	ax.plot(np.sum(X, axis = 0))

	plt.show()


def plot_expert_distances (distances, N, C, R):
	"""
	Plot the Euclidean distance of each caching policy expert to the BSH cache configuration.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Expert cache configuration distance to BSH over time\n[N = {N}, C = {C}]')

	ax.set_ylabel('Euclidean distance')
	ax.set_xlabel('Time slot')

	for i, r in enumerate(R):
		ax.plot(distances[i], label = f'OGA [{r}]')

	plt.legend()
	plt.show()


def plot_expert_utilities (utilities, T, N, C, R):
	"""
	Plot the utility progression of each caching policy expert.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Expert utility over time\n[N = {N}, C = {C}]')

	ax.set_ylabel('Average utility')
	ax.set_xlabel('Time slot')

	# Generate list of time slots to be used in deriving average utility per time slot
	time_slots = np.arange(1, T + 1)

	for i, r in enumerate(R):
		ax.plot(utilities[i] / time_slots, label = f'OGA [{r}]')

	plt.legend()
	plt.show()


def plot_meta_learner_weights (weights, R, L):
	"""
	Plot the caching policy expert weights of the meta-learner.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Meta-learner expert weights over time\n[L = {L}]')

	ax.set_ylabel('Expert weight')
	ax.set_xlabel('Time slot')

	for i, r in enumerate(R):
		ax.plot(weights[:, i], label = f'OGA [{r}]')

	plt.legend()
	plt.show()


def plot_meta_learner_utility (utility, T, N, C, L):
	"""
	Plot the utility progression of the meta-learner.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Meta-learner utility over time\n[N = {N}, C = {C}, L = {L}]')

	ax.set_ylabel('Average utility')
	ax.set_xlabel('Time slot')

	# Generate list of time slots to be used in deriving average utility per time slot
	time_slots = np.arange(1, T + 1)

	ax.plot(utility / time_slots)

	plt.show()

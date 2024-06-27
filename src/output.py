import numpy as np
import matplotlib.pyplot as plt


def save_request_matrix (X, file_name):
	"""
	Save the request matrix (X) to disk.
	"""
	with open(f'./results/{file_name}.npy', 'wb') as f:
		np.save(f, X)


def load_request_matrix (file_name):
	"""
	Load the request matrix (X) from disk.
	"""
	with open(f'./results/{file_name}.npy', 'rb') as f:
		X = np.load(f)

	return X


def save_results (utility, distance, file_name):
	"""
	Save the utility progression and cache distances of a caching policy to disk.
	"""
	with open(f'./results/utility/{file_name}.npy', 'wb') as f:
		np.save(f, utility)

	with open(f'./results/cache_distance/{file_name}.npy', 'wb') as f:
		np.save(f, distance)


def load_results (file_name):
	"""
	Load the utility progression and cache distances of a caching policy from disk.
	"""
	with open(f'./results/utility/{file_name}.npy', 'rb') as f:
		utility = np.load(f)

	with open(f'./results/cache_distance/{file_name}.npy', 'rb') as f:
		distance = np.load(f)

	return (utility, distance)


def plot_request_distribution (X, T):
	"""
	Plot the request distribution (X) over the horizon (T).
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'File request distribution\n[T = {T}]')

	ax.set_ylabel('Number of requests')
	ax.set_xlabel('File')

	# Plot the total number of requests per file over horizon T (request distribution)
	ax.plot(np.sum(X, axis = 0))

	plt.savefig('./results/plots/request_distribution.png', dpi = 300)
	plt.show()


def plot_expert_distances (distances, N, C, K):
	"""
	Plot the Euclidean distance of each caching expert to the BSCH cache configuration.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Expert cache configuration distance to BSCH over time\n[N = {N}, C = {C}]')

	ax.set_ylabel('Euclidean distance')
	ax.set_xlabel('Time slot')

	for i, k in enumerate(K):
		ax.plot(distances[i], label = f'OGA [{k}]')

	plt.legend()
	plt.savefig('./results/plots/expert_distances.png', dpi = 300)
	plt.show()


def plot_expert_utilities (utilities, T, N, C, K):
	"""
	Plot the utility progression of each caching expert.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Expert utility over time\n[N = {N}, C = {C}]')

	ax.set_ylabel('Average utility')
	ax.set_xlabel('Time slot')

	# Generate list of time slots to be used in deriving the average utility per time slot
	time_slots = np.arange(1, T + 1)

	for i, k in enumerate(K):
		ax.plot(utilities[i] / time_slots, label = f'OGA [{k}]')

	plt.legend()
	plt.savefig('./results/plots/expert_utilities.png', dpi = 300)
	plt.show()


def plot_meta_learner_weights (weights, N, C, K):
	"""
	Plot the caching expert weights progression of the meta-learner.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Meta-learner expert weights over time\n[N = {N}, C = {C}]')

	ax.set_ylabel('Expert weight')
	ax.set_xlabel('Time slot')

	for i, k in enumerate(K):
		ax.plot(weights[:, i], label = f'OGA [{k}]')

	plt.legend()
	plt.savefig('./results/plots/meta_learner_weights.png', dpi = 300)
	plt.show()


def plot_meta_learner_utility (utility, T, N, C):
	"""
	Plot the utility progression of the meta-learner.
	"""
	fig, ax = plt.subplots()
	fig.suptitle(f'Meta-learner utility over time\n[N = {N}, C = {C}]')

	ax.set_ylabel('Average utility')
	ax.set_xlabel('Time slot')

	# Generate list of time slots to be used in deriving the average utility per time slot
	time_slots = np.arange(1, T + 1)

	ax.plot(utility / time_slots, label = '$\\sigma^*$')

	plt.savefig('./results/plots/meta_learner_utility.png', dpi = 300)
	plt.show()

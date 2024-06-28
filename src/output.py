import numpy as np
import os
import matplotlib.pyplot as plt


def save (data, file_path):
	"""
	Save the data to disk at the specified file path.
	"""

	# Extract the directory path from the file path
	directory = os.path.dirname(file_path)

	# Create the output directories if they are non-existent
	if not os.path.exists(directory):
		os.makedirs(directory)

	with open(f'{file_path}.npy', 'wb') as f:
		np.save(f, data)


def load (file_path):
	"""
	Load the data from disk at the specified file path.
	"""
	with open(f'{file_path}.npy', 'rb') as f:
		data = np.load(f)

	return np.asarray(data)


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

	# Create the output directory if it is non-existent
	if not os.path.exists('./data/plots'):
		os.makedirs('./data/plots')

	plt.savefig('./data/plots/request_distribution.png', dpi = 300)
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
		ax.plot(utilities[i] / time_slots, label = f'OGA [{k:.2f}]')

	# Create the output directory if it is non-existent
	if not os.path.exists('./data/plots'):
		os.makedirs('./data/plots')

	plt.legend()
	plt.savefig('./data/plots/expert_utilities.png', dpi = 300)
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
		ax.plot(weights[:, i], label = f'OGA [{k:.2f}]')

	# Create the output directory if it is non-existent
	if not os.path.exists('./data/plots'):
		os.makedirs('./data/plots')

	plt.legend()
	plt.savefig('./data/plots/meta_learner_weights.png', dpi = 300)
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

	# Create the output directory if it is non-existent
	if not os.path.exists('./data/plots'):
		os.makedirs('./data/plots')

	plt.savefig('./data/plots/meta_learner_utility.png', dpi = 300)
	plt.show()

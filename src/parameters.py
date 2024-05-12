import generate

# The number of time slots, and consequently the number of requests or iterations (horizon)
T = 50000

# The size of the system's library of files and cache, where C should be (significantly) smaller than N
N = 5000
C = 500

# T-by-N matrix: at each timeslot t a single entry is set to True (file request)
X = generate.random_X_zipfian(T, N, alpha = 0.8)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)

# Learning rates to be used in online gradient ascent algorithm (computed dynamically for None entries)
R = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

# Learning rate of the meta learner
L = 0.2


# Assert that the input parameters are within valid ranges
assert T > 0, 'The number of time slots T must be positive'
assert N > 0, 'The system library size N must be positive'
assert C > 0 and C < N, 'The cache size C must be within range [1 .. N - 1]'
assert R, 'The list of learning rates R must not be empty'
assert L > 0, 'The meta learner learning rate L must be positive'

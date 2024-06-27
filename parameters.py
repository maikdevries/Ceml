from src import generate

# The number of time slots, and consequently the number of requests or iterations (horizon)
T = 200000

# The size of the system's library of files and cache, where C should be (significantly) smaller than N
N = 2500
C = 250

# T-by-N matrix: at each timeslot t a single entry is set to True (file request)
X = generate.random_X_zipfian(T, N, alpha = 0.8)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)

# Learning rates to be used in online gradient ascent (caching agent) algorithm (computed dynamically for None entries)
R = [0.05, 0.1, 0.3, 1.0]


# Assert that the input parameters are within valid ranges
assert T > 0, 'The number of time slots T must be positive'
assert N > 0, 'The system library size N must be positive'
assert C > 0 and C < N, 'The cache size C must be within range [1 .. N - 1]'
assert all(sum(x) == 1 for x in X), 'Each request vector in X must have exactly one element set to True'
assert R, 'The list of caching agent learning rates R must not be empty'
assert all(r > 0 for r in R), 'The list of caching agent learning rates R must be positive'

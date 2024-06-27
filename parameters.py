from src import generate

# The number of time slots and consequently, the number of requests (horizon)
T = 200000

# The size of the system's library of files and cache, where C should be (significantly) smaller than N
N = 2500
C = 250

# T-by-N matrix: for each time slot t, a single element of vector x is set to True (file request)
X = generate.X_zipfian(T, N, zeta = 0.8)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.W_uniform(N)

# Distinct learning rates to be used in the online gradient ascent algorithm (caching experts)
K = [0.05, 0.1, 0.3, 1.0]


# Assert that the input parameters are within valid ranges
assert T > 0, 'The number of time slots T must be positive'
assert N > 0, 'The system library size N must be positive'
assert C > 0 and C < N, 'The cache size C must be within range [1 .. N - 1]'
assert all(sum(x) == 1 for x in X), 'Each request vector in X must have exactly one element set to True'
assert K, 'The list of caching expert learning rates K must not be empty'
assert all(k > 0 for k in K), 'The list of caching expert learning rates K must be positive'

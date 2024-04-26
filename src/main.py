import generate
import benchmark


# The number of time slots, and consequently the number of requests or iterations (horizon)
T = 1000

# The size of the system's library of files and cache, where C should be (significantly) smaller than N
N = 10000
C = 100

# T-by-N matrix: at each timeslot t a single entry is set to True (file request)
X = generate.random_X(T, N)

# N-dimensional vector: at each timeslot t all entries will sum up to at most C (cache configuration)
Y = generate.random_Y(N, C)

# N-dimensional vector: static file-dependent utility vector (utility weights)
W = generate.uniform_weights(N)


# TODO: assert C is within range [0 .. N]
if __name__ == '__main__':

	# Calculate sum of utility for various caching policies
	OGA_utility = benchmark.calc_utility_OGA(X, Y, W, T, N, C)
	hindsight_utility = benchmark.calc_utility_hindsight(X, W, C)

	print("Utility accumulated by OGA policy:", OGA_utility)
	print("Utility accumulated by best static cache configuration in hindsight:", hindsight_utility)
	print("Regret achieved by OGA policy:", (hindsight_utility - OGA_utility))

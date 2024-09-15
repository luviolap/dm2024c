import random
from sympy import sieve, prime

def generate_prime_seeds(initial_seed: int, num_seeds: int, min_prime: int = 100000, max_prime: int = 1000000) -> list:
    """
    Generate a list of random prime number seeds.
    
    Args:
    initial_seed (int): The initial seed for random number generation.
    num_seeds (int): The number of prime seeds to generate.
    min_prime (int): The minimum value for prime numbers (default: 100000).
    max_prime (int): The maximum value for prime numbers (default: 1000000).
    
    Returns:
    list: A list of randomly selected prime numbers to be used as seeds.
    """
    # Generate prime numbers in the specified range
    sieve.extend(max_prime)
    primes = list(sieve.primerange(min_prime, max_prime))
    
    # Set the initial random seed
    random.seed(initial_seed)
    
    # Randomly select prime numbers to use as seeds
    seeds = random.sample(primes, num_seeds)
    
    return seeds

# Example usage
initial_seed = 102191
num_seeds = 20

seeds = generate_prime_seeds(initial_seed, num_seeds)
print(seeds)
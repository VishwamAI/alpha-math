import sympy

def prime_factorization(n):
    """Return the prime factorization of a number."""
    factors = sympy.factorint(n)
    return factors

def greatest_common_divisor(a, b):
    """Calculate the greatest common divisor of two numbers."""
    return sympy.gcd(a, b)

def least_common_multiple(a, b):
    """Calculate the least common multiple of two numbers."""
    return sympy.lcm(a, b)

def euler_totient(n):
    """Calculate Euler's totient function for a number."""
    return sympy.totient(n)

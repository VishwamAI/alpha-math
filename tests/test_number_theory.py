import pytest
from alphamath.number_theory import number_theory

def test_is_prime():
    # Test the is_prime function with a prime number
    assert number_theory.is_prime(5) is True

def test_is_prime_non_prime():
    # Test the is_prime function with a non-prime number
    assert number_theory.is_prime(4) is False

def test_prime_factorization():
    # Test the prime_factorization function
    assert number_theory.prime_factorization(12) == {2: 2, 3: 1}

def test_gcd():
    # Test the greatest_common_divisor function
    assert number_theory.greatest_common_divisor(48, 18) == 6

def test_lcm():
    # Test the least_common_multiple function
    assert number_theory.least_common_multiple(12, 18) == 36

def test_euler_totient():
    # Test the euler_totient function
    assert number_theory.euler_totient(36) == 12

# More tests will be added here to cover other functions and edge cases

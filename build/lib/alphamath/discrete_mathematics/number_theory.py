import math
from sympy import prime, factorint, gcd, lcm, totient

def is_prime(n):
    """
    Check if a number is prime.

    :param n: An integer to check for primality
    :return: Boolean indicating whether the number is prime
    """
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def prime_factorization(n):
    """
    Compute the prime factorization of a number.

    :param n: An integer to factorize
    :return: Dictionary with prime factors as keys and their multiplicities as values
    """
    return dict(factorint(n))

def greatest_common_divisor(a, b):
    """
    Calculate the Greatest Common Divisor of two numbers.

    :param a: First integer
    :param b: Second integer
    :return: The Greatest Common Divisor of a and b
    """
    return gcd(a, b)

def least_common_multiple(a, b):
    """
    Calculate the Least Common Multiple of two numbers.

    :param a: First integer
    :param b: Second integer
    :return: The Least Common Multiple of a and b
    """
    return lcm(a, b)

def euler_totient(n):
    """
    Calculate Euler's totient function value for n.

    :param n: A positive integer
    :return: The number of integers k in the range 1 <= k <= n for which gcd(n, k) = 1
    """
    return totient(n)

def modular_exponentiation(base, exponent, modulus):
    """
    Perform modular exponentiation efficiently.

    :param base: Base of the exponentiation
    :param exponent: Exponent
    :param modulus: Modulus
    :return: (base^exponent) % modulus
    """
    return pow(base, exponent, modulus)

def extended_euclidean(a, b):
    """
    Implement the Extended Euclidean Algorithm.

    :param a: First integer
    :param b: Second integer
    :return: Tuple (gcd, x, y) where gcd is the greatest common divisor and ax + by = gcd
    """
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_euclidean(b % a, a)
        return gcd, y - (b // a) * x, x

def chinese_remainder_theorem(remainders, moduli):
    """
    Implement the Chinese Remainder Theorem.

    :param remainders: List of remainders
    :param moduli: List of moduli
    :return: Solution to the system of congruences
    """
    total = 0
    product = math.prod(moduli)
    for remainder, modulus in zip(remainders, moduli):
        p = product // modulus
        total += remainder * pow(p, -1, modulus) * p
    return total % product

# Example usage
if __name__ == "__main__":
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"Prime factorization of 84: {prime_factorization(84)}")
    print(f"GCD of 48 and 18: {greatest_common_divisor(48, 18)}")
    print(f"LCM of 12 and 18: {least_common_multiple(12, 18)}")
    print(f"Euler's totient of 36: {euler_totient(36)}")
    print(f"3^7 mod 13: {modular_exponentiation(3, 7, 13)}")
    print(f"Extended Euclidean for 48 and 18: {extended_euclidean(48, 18)}")
    print(f"Chinese Remainder Theorem for x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7): {chinese_remainder_theorem([2, 3, 2], [3, 5, 7])}")

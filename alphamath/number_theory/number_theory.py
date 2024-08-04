import math

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

def gcd(a, b):
    """
    Calculate the Greatest Common Divisor of a and b.

    :param a: First integer
    :param b: Second integer
    :return: The Greatest Common Divisor of a and b
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """
    Calculate the Least Common Multiple of a and b.

    :param a: First integer
    :param b: Second integer
    :return: The Least Common Multiple of a and b
    """
    return abs(a * b) // gcd(a, b)

def calculate_totient(n):
    """
    Calculate Euler's totient function value for n.

    :param n: A positive integer
    :return: The number of integers k in the range 1 <= k <= n for which gcd(n, k) = 1
    """
    result = n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            while n % i == 0:
                n //= i
            result *= (1 - 1/i)
    if n > 1:
        result *= (1 - 1/n)
    return int(result)

# Example usage
if __name__ == "__main__":
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"GCD of 48 and 18: {gcd(48, 18)}")
    print(f"LCM of 12 and 18: {lcm(12, 18)}")
    print(f"Euler's totient of 36: {calculate_totient(36)}")

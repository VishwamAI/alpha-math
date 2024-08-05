import math
from sympy import factorial, binomial

def calculate_permutations(n, r):
    """
    Calculate the number of permutations of n items taken r at a time.

    :param n: Total number of items
    :param r: Number of items being arranged
    :return: Number of permutations
    """
    return math.perm(n, r)

def calculate_combinations(n, r):
    """
    Calculate the number of combinations of n items taken r at a time.

    :param n: Total number of items
    :param r: Number of items being chosen
    :return: Number of combinations
    """
    return math.comb(n, r)

def calculate_factorial(n):
    """
    Calculate the factorial of a non-negative integer.

    :param n: Non-negative integer
    :return: Factorial of n
    """
    return factorial(n)

def stirling_number_second_kind(n, k):
    """
    Calculate the Stirling number of the second kind.

    :param n: Number of items
    :param k: Number of non-empty subsets
    :return: Stirling number of the second kind
    """
    return sum((-1)**(k-i) * binomial(k, i) * i**n for i in range(k+1)) // factorial(k)

def bell_number(n):
    """
    Calculate the Bell number.

    :param n: Non-negative integer
    :return: nth Bell number
    """
    return sum(stirling_number_second_kind(n, k) for k in range(n+1))

def catalan_number(n):
    """
    Calculate the nth Catalan number.

    :param n: Non-negative integer
    :return: nth Catalan number
    """
    return binomial(2*n, n) // (n + 1)

def derangement(n):
    """
    Calculate the number of derangements of n items.

    :param n: Number of items
    :return: Number of derangements
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    return (n - 1) * (derangement(n - 1) + derangement(n - 2))

def partition_number(n):
    """
    Calculate the partition number of n.

    :param n: Non-negative integer
    :return: Number of ways to partition n
    """
    partitions = [1] + [0] * n
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            partitions[j] += partitions[j - i]
    return partitions[n]

# Example usage
if __name__ == "__main__":
    print(f"P(5,3) = {calculate_permutations(5, 3)}")
    print(f"C(5,3) = {calculate_combinations(5, 3)}")
    print(f"5! = {calculate_factorial(5)}")
    print(f"Stirling2(4,2) = {stirling_number_second_kind(4, 2)}")
    print(f"Bell(4) = {bell_number(4)}")
    print(f"Catalan(4) = {catalan_number(4)}")
    print(f"Derangement(4) = {derangement(4)}")
    print(f"Partition(5) = {partition_number(5)}")

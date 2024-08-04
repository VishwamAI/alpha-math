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

# Example usage
if __name__ == "__main__":
    # Calculate permutations
    n, r = 5, 3
    perm_result = calculate_permutations(n, r)
    print(f"Number of permutations P({n},{r}): {perm_result}")

    # Calculate combinations
    comb_result = calculate_combinations(n, r)
    print(f"Number of combinations C({n},{r}): {comb_result}")

    # Calculate factorial
    fact_n = 5
    fact_result = calculate_factorial(fact_n)
    print(f"Factorial of {fact_n}: {fact_result}")

    # Calculate Stirling number of the second kind
    stirling_n, stirling_k = 5, 3
    stirling_result = stirling_number_second_kind(stirling_n, stirling_k)
    print(f"Stirling number of the second kind S({stirling_n},{stirling_k}): {stirling_result}")

    # Calculate Bell number
    bell_n = 5
    bell_result = bell_number(bell_n)
    print(f"Bell number B({bell_n}): {bell_result}")

    # Calculate Catalan number
    catalan_n = 5
    catalan_result = catalan_number(catalan_n)
    print(f"Catalan number C({catalan_n}): {catalan_result}")

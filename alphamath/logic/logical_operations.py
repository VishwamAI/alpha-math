import sympy as sp

def logical_and(p, q):
    """
    Perform logical AND operation.

    :param p: First proposition
    :param q: Second proposition
    :return: Result of p AND q
    """
    return sp.And(p, q)

def logical_or(p, q):
    """
    Perform logical OR operation.

    :param p: First proposition
    :param q: Second proposition
    :return: Result of p OR q
    """
    return sp.Or(p, q)

def logical_not(p):
    """
    Perform logical NOT operation.

    :param p: Proposition to negate
    :return: Negation of p
    """
    return sp.Not(p)

def logical_implication(p, q):
    """
    Perform logical implication operation.

    :param p: Antecedent
    :param q: Consequent
    :return: Result of p IMPLIES q
    """
    return sp.Implies(p, q)

def logical_equivalence(p, q):
    """
    Perform logical equivalence operation.

    :param p: First proposition
    :param q: Second proposition
    :return: Result of p EQUIVALENT TO q
    """
    return sp.Equivalent(p, q)

def logical_xor(p, q):
    """
    Perform logical XOR operation.

    :param p: First proposition
    :param q: Second proposition
    :return: Result of p XOR q
    """
    return sp.Xor(p, q)

# Example usage
if __name__ == "__main__":
    p, q = sp.symbols('p q')
    print(f"p AND q: {logical_and(p, q)}")
    print(f"p OR q: {logical_or(p, q)}")
    print(f"NOT p: {logical_not(p)}")
    print(f"p IMPLIES q: {logical_implication(p, q)}")
    print(f"p EQUIVALENT TO q: {logical_equivalence(p, q)}")
    print(f"p XOR q: {logical_xor(p, q)}")

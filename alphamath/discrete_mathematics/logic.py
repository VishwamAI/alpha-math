import sympy as sp

def negation(proposition):
    """
    Negate a logical proposition.

    :param proposition: A SymPy Boolean expression
    :return: Negation of the proposition
    """
    return ~proposition

def conjunction(prop1, prop2):
    """
    Perform logical conjunction (AND) of two propositions.

    :param prop1: First SymPy Boolean expression
    :param prop2: Second SymPy Boolean expression
    :return: Conjunction of the propositions
    """
    return prop1 & prop2

def disjunction(prop1, prop2):
    """
    Perform logical disjunction (OR) of two propositions.

    :param prop1: First SymPy Boolean expression
    :param prop2: Second SymPy Boolean expression
    :return: Disjunction of the propositions
    """
    return prop1 | prop2

def implication(antecedent, consequent):
    """
    Perform logical implication.

    :param antecedent: SymPy Boolean expression representing the antecedent
    :param consequent: SymPy Boolean expression representing the consequent
    :return: Implication of antecedent and consequent
    """
    return sp.Implies(antecedent, consequent)

def equivalence(prop1, prop2):
    """
    Check logical equivalence of two propositions.

    :param prop1: First SymPy Boolean expression
    :param prop2: Second SymPy Boolean expression
    :return: Boolean indicating if the propositions are equivalent
    """
    return sp.Equivalent(prop1, prop2)

def truth_table(expression, variables):
    """
    Generate a truth table for a given logical expression.

    :param expression: SymPy Boolean expression
    :param variables: List of SymPy Symbol objects representing variables in the expression
    :return: List of tuples representing rows in the truth table
    """
    table = []
    for values in sp.itertools.product([True, False], repeat=len(variables)):
        row = dict(zip(variables, values))
        result = expression.subs(row)
        table.append((*values, result))
    return table

def is_tautology(expression):
    """
    Check if a given logical expression is a tautology.

    :param expression: SymPy Boolean expression
    :return: Boolean indicating if the expression is a tautology
    """
    return expression.is_tautology()

def is_contradiction(expression):
    """
    Check if a given logical expression is a contradiction.

    :param expression: SymPy Boolean expression
    :return: Boolean indicating if the expression is a contradiction
    """
    return expression.is_contradiction()

# Example usage
if __name__ == "__main__":
    p, q = sp.symbols('p q')
    expr = sp.Implies(p, q)

    print("Negation of p:", negation(p))
    print("Conjunction of p and q:", conjunction(p, q))
    print("Disjunction of p and q:", disjunction(p, q))
    print("Implication p -> q:", implication(p, q))
    print("Equivalence of p and q:", equivalence(p, q))

    print("\nTruth table for p -> q:")
    for row in truth_table(expr, [p, q]):
        print(row)

    print("\nIs p -> q a tautology?", is_tautology(expr))
    print("Is p & ~p a contradiction?", is_contradiction(p & ~p))

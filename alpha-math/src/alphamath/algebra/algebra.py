import sympy as sp
import random

def solve_equation(equation):
    """
    Solve a given algebraic equation.

    :param equation: string representing the equation to solve
    :return: solution(s) to the equation
    """
    try:
        return sp.solve(equation)
    except Exception as e:
        return f"Error solving equation: {str(e)}"

def factor_polynomial(polynomial):
    """
    Factor a given polynomial.

    :param polynomial: string representing the polynomial to factor
    :return: factored form of the polynomial
    """
    try:
        return sp.factor(polynomial)
    except Exception as e:
        return f"Error factoring polynomial: {str(e)}"

def simplify_expression(expression):
    """
    Simplify a given algebraic expression.

    :param expression: string representing the expression to simplify
    :return: simplified form of the expression
    """
    try:
        return sp.simplify(expression)
    except Exception as e:
        return f"Error simplifying expression: {str(e)}"

def expand_expression(expression):
    """
    Expand a given algebraic expression.

    :param expression: string representing the expression to expand
    :return: expanded form of the expression
    """
    try:
        return sp.expand(expression)
    except Exception as e:
        return f"Error expanding expression: {str(e)}"

def polynomial_roots(coefficients):
    """
    Find the roots of a polynomial given its coefficients.

    :param coefficients: list of coefficients (highest degree first)
    :return: list of roots
    """
    try:
        x = sp.Symbol('x')
        polynomial = sum(coeff * x**i for i, coeff in enumerate(reversed(coefficients)))
        return sp.solve(polynomial)
    except Exception as e:
        return f"Error finding polynomial roots: {str(e)}"

def _solve_linear_system(equations):
    """
    Solve a system of linear equations.

    :param equations: list of strings representing linear equations
    :return: solution to the system of equations
    """
    try:
        return sp.solve(equations)
    except Exception as e:
        return f"Error solving linear system: {str(e)}"

def sequence_next_term(sequence):
    """
    Predict the next term in a given sequence.

    :param sequence: list of numbers representing the sequence
    :return: predicted next term
    """
    try:
        n = len(sequence)
        x = sp.Symbol('x')
        polynomial = sp.interpolate(list(zip(range(n), sequence)), x)
        return polynomial.subs(x, n)
    except Exception as e:
        return f"Error predicting next term: {str(e)}"

def sequence_nth_term(sequence, n):
    """
    Find the nth term of a given sequence.

    :param sequence: list of numbers representing the sequence
    :param n: position of the term to find (1-indexed)
    :return: nth term of the sequence
    """
    try:
        x = sp.Symbol('x')
        polynomial = sp.interpolate(list(zip(range(1, len(sequence) + 1), sequence)), x)
        return polynomial.subs(x, n)
    except Exception as e:
        return f"Error finding nth term: {str(e)}"

def generate_algebra_problem(difficulty='medium'):
    """
    Generate an algebra problem based on the specified difficulty.

    :param difficulty: string representing the difficulty level ('easy', 'medium', 'hard')
    :return: tuple containing the problem statement and its solution
    """
    if difficulty == 'easy':
        a, b = random.randint(1, 10), random.randint(1, 10)
        x = sp.Symbol('x')
        equation = sp.Eq(a*x + b, random.randint(1, 50))
        problem = f"Solve the equation: {equation}"
        solution = sp.solve(equation)[0]
    elif difficulty == 'medium':
        a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
        x = sp.Symbol('x')
        equation = sp.Eq(a*x**2 + b*x + c, 0)
        problem = f"Find the roots of the quadratic equation: {equation}"
        solution = sp.solve(equation)
    elif difficulty == 'hard':
        sequence = [random.randint(1, 20) for _ in range(5)]
        problem = f"Find the next term in the sequence: {sequence}"
        solution = sequence_next_term(sequence)
    else:
        raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")

    return problem, solution

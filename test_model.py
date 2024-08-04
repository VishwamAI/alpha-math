import sys
import os
import pytest
from sympy import sympify

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alphamath.model import solve_equation, calculate_expression, evaluate_function, factor_check

def test_solve_equation():
    equation_with_solve = "Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r"
    result_with_solve = solve_equation(equation_with_solve)
    assert abs(result_with_solve - 4) < 1e-6, f"Expected 4, but got {result_with_solve}"

    equation_without_solve = "-42*r + 27*c = -1167 and 130*r + 4*c = 372"
    result_without_solve = solve_equation(f"Solve {equation_without_solve} for r")
    assert abs(result_without_solve - 4) < 1e-6, f"Expected 4, but got {result_without_solve}"

def test_calculate_expression():
    expression = "-841880142.544 + 411127"
    result = calculate_expression(expression)
    expected = -841469015.544
    assert abs(result - expected) < 1e-6, f"Expected {expected}, but got {result}"

def test_evaluate_function():
    functions = {
        'x': 'x(g) = 9*g + 1',
        'q': 'q(c) = 2*c + 1',
        'f': 'f(i) = 3*i - 39',
        'w': 'w(j) = q(x(j))'
    }
    composite = "f(w(a))"

    result = evaluate_function(composite, functions, 'a')
    expected = sympify("54*a - 30")
    assert result == expected, f"Expected {expected}, but got {result}"

def test_factor_check():
    assert factor_check(10, 2) == True, "2 should be a factor of 10"
    assert factor_check(10, 3) == False, "3 should not be a factor of 10"
    assert factor_check(0, 5) == True, "0 should be divisible by any non-zero number"
    assert factor_check(7, 0) == False, "0 should not be a factor of any number"
    assert factor_check(-12, 3) == True, "3 should be a factor of -12"
    assert factor_check(-12, -4) == True, "-4 should be a factor of -12"

    with pytest.raises(ValueError):
        factor_check("not a number", 2)

def test_evaluate_function_edge_cases():
    functions = {
        'f': 'f(x) = x^2',
        'g': 'g(x) = 2*x + 1',
        'h': 'h(x) = f(g(x))',
        'j': 'j(x) = 3*x + 2',
        'k': 'k(x) = j(h(x))'
    }

    result = evaluate_function('h(3)', functions, '3')
    expected = sympify("49")
    assert result == expected, f"Expected {expected}, but got {result}"

    result = evaluate_function('k(2)', functions, '2')
    expected = sympify("147")
    assert result == expected, f"Expected {expected}, but got {result}"

    result = evaluate_function('diff(h(x), x)', functions, 'x')
    expected = sympify("4*x + 2")
    assert result == expected, f"Expected {expected}, but got {result}"

    with pytest.raises(ValueError):
        evaluate_function('nonexistent(x)', functions, '3')

    with pytest.raises(ValueError):
        evaluate_function('f(x, y)', functions, 'x')

def test_derivative():
    functions = {
        'u': 'u(n) = -n**3 - n**2',
        'e': 'e(c) = -2*c**3 + c',
        'l': 'l(j) = -118*e(j) + 54*u(j)'
    }
    result = evaluate_function('diff(l(a), a)', functions, 'a')
    expected = sympify("546*a**2 - 108*a - 118")
    assert result == expected, f"Expected {expected}, but got {result}"

def test_probability():
    total_letters = 12
    q_count = 4
    k_count = 8
    # Probability of selecting 2 'Q's and 1 'K' in that order
    prob = (q_count / total_letters) * ((q_count - 1) / (total_letters - 1)) * (k_count / (total_letters - 2))
    expected = 4/165  # Correct probability
    assert abs(prob - expected) < 1e-6, f"Expected {expected}, but got {prob}"

    # Additional test case for probability calculation
    def calculate_probability(total, q, k):
        return (q / total) * ((q - 1) / (total - 1)) * (k / (total - 2))

    assert abs(calculate_probability(12, 4, 8) - 4/165) < 1e-6, "Probability calculation is incorrect"

    # Test with different values
    assert abs(calculate_probability(52, 4, 4) - 24/5525) < 1e-6, "Probability calculation is incorrect for standard deck"

    # Test with edge cases
    assert calculate_probability(3, 1, 2) == 0, "Probability should be 0 when not enough letters"
    assert calculate_probability(3, 3, 0) == 0, "Probability should be 0 when no 'K's available"

if __name__ == "__main__":
    pytest.main([__file__])

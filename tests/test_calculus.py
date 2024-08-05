import pytest
from alphamath.calculus import calculus

def test_calculate_derivative():
    # Test the calculate_derivative function with a simple function
    assert calculus.calculate_derivative('x^2', 'x') == '2*x'

def test_calculate_integral():
    # Test the calculate_integral function with a simple function
    assert calculus.calculate_integral('x', 'x') == '0.5*x^2'

def test_calculate_limit():
    # Test the calculate_limit function
    assert calculus.calculate_limit('(x^2 - 1)/(x - 1)', 'x', 1) == 2

def test_calculate_series_expansion():
    # Test the calculate_series_expansion function
    assert calculus.calculate_series_expansion('sin(x)', 'x', 0, 3) == 'x - x^3/6 + O(x^5)'

# More tests will be added here to cover other functions and edge cases

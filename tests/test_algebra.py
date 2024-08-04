"""
Test suite for the algebra module of the alpha-math library.
"""

import pytest
from alphamath.algebra import algebra

def test_solve_linear_equation():
    # Test the solve_linear_equation function with a simple equation
    assert algebra.solve_linear_equation('x + 2 = 5') == 3

def test_solve_quadratic_equation():
    # Test the solve_quadratic_equation function with a simple quadratic equation
    roots = algebra.solve_quadratic_equation('x^2 - 4x + 4')
    assert set(roots) == {2}

# More tests will be added here to cover other functions and edge cases

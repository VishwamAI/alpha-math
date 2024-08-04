import numpy as np

def bisection_method(function, a, b, tolerance=1e-6, max_iterations=100):
    """
    Find a root of a function using the bisection method.

    Parameters:
    function (callable): The function to find the root of
    a (float): Left endpoint of the interval
    b (float): Right endpoint of the interval
    tolerance (float): The desired accuracy (default: 1e-6)
    max_iterations (int): Maximum number of iterations (default: 100)

    Returns:
    float: Approximate root of the function
    """
    if function(a) * function(b) >= 0:
        raise ValueError("Function values at interval endpoints must have opposite signs")

    for _ in range(max_iterations):
        c = (a + b) / 2
        if abs(function(c)) < tolerance:
            return c
        if function(c) * function(a) < 0:
            b = c
        else:
            a = c

    raise RuntimeError(f"Method failed to converge within {max_iterations} iterations")

def newtons_method(function, derivative, x0, tolerance=1e-6, max_iterations=100):
    """
    Find a root of a function using Newton's method.

    Parameters:
    function (callable): The function to find the root of
    derivative (callable): The derivative of the function
    x0 (float): Initial guess
    tolerance (float): The desired accuracy (default: 1e-6)
    max_iterations (int): Maximum number of iterations (default: 100)

    Returns:
    float: Approximate root of the function
    """
    x = x0
    for _ in range(max_iterations):
        fx = function(x)
        if abs(fx) < tolerance:
            return x
        dfx = derivative(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        x = x - fx / dfx
    raise RuntimeError(f"Method failed to converge within {max_iterations} iterations")

def secant_method(function, x0, x1, tolerance=1e-6, max_iterations=100):
    """
    Find a root of a function using the secant method.

    Parameters:
    function (callable): The function to find the root of
    x0 (float): First initial guess
    x1 (float): Second initial guess
    tolerance (float): The desired accuracy (default: 1e-6)
    max_iterations (int): Maximum number of iterations (default: 100)

    Returns:
    float: Approximate root of the function
    """
    for _ in range(max_iterations):
        fx0 = function(x0)
        fx1 = function(x1)
        if abs(fx1) < tolerance:
            return x1
        if fx0 == fx1:
            raise ValueError("Division by zero in secant method.")
        x = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x
    raise RuntimeError(f"Method failed to converge within {max_iterations} iterations")

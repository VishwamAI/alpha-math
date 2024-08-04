import numpy as np

def forward_difference(function, x, h):
    """
    Implement the forward difference method for approximating the derivative of a function at a point x with step size h.

    :param function: The function to differentiate
    :param x: The point at which to approximate the derivative
    :param h: The step size
    :return: Approximation of the derivative
    """
    return (function(x + h) - function(x)) / h

def backward_difference(function, x, h):
    """
    Implement the backward difference method for approximating the derivative of a function at a point x with step size h.

    :param function: The function to differentiate
    :param x: The point at which to approximate the derivative
    :param h: The step size
    :return: Approximation of the derivative
    """
    return (function(x) - function(x - h)) / h

def central_difference(function, x, h):
    """
    Implement the central difference method for approximating the derivative of a function at a point x with step size h.

    :param function: The function to differentiate
    :param x: The point at which to approximate the derivative
    :param h: The step size
    :return: Approximation of the derivative
    """
    return (function(x + h) - function(x - h)) / (2 * h)

# Example usage
if __name__ == "__main__":
    def f(x):
        return x**2

    x = 2
    h = 0.001

    print(f"Forward difference: {forward_difference(f, x, h)}")
    print(f"Backward difference: {backward_difference(f, x, h)}")
    print(f"Central difference: {central_difference(f, x, h)}")

import numpy as np

def trapezoidal_rule(function, a, b, n):
    """
    Implement the trapezoidal rule for approximating the integral of a function over an interval [a, b] using n subintervals.

    :param function: The function to integrate
    :param a: Lower bound of the interval
    :param b: Upper bound of the interval
    :param n: Number of subintervals
    :return: Approximation of the integral
    """
    x = np.linspace(a, b, n+1)
    y = function(x)
    return (b - a) / (2 * n) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

def simpsons_rule(function, a, b, n):
    """
    Implement Simpson's rule for approximating the integral of a function over an interval [a, b] using n subintervals.

    :param function: The function to integrate
    :param a: Lower bound of the interval
    :param b: Upper bound of the interval
    :param n: Number of subintervals (must be even)
    :return: Approximation of the integral
    """
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")

    x = np.linspace(a, b, n+1)
    y = function(x)
    return (b - a) / (3 * n) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

def gaussian_quadrature(function, a, b, n):
    """
    Implement Gaussian quadrature for approximating the integral of a function over an interval [a, b] using n points.

    :param function: The function to integrate
    :param a: Lower bound of the interval
    :param b: Upper bound of the interval
    :param n: Number of points
    :return: Approximation of the integral
    """
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * function(t))

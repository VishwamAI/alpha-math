import sympy as sp
from sympy import limit, diff, integrate, summation, oo

def calculate_limit(expression, variable, point):
    """
    Calculate the limit of an expression as the variable approaches a point.

    :param expression: The mathematical expression (SymPy expression)
    :param variable: The variable of the limit (SymPy symbol)
    :param point: The point the variable approaches (number or SymPy expression)
    :return: The limit of the expression
    """
    return limit(expression, variable, point)

def calculate_derivative(expression, variable, order=1):
    """
    Calculate the derivative of an expression with respect to a variable.

    :param expression: The mathematical expression (SymPy expression)
    :param variable: The variable to differentiate with respect to (SymPy symbol)
    :param order: The order of the derivative (default is 1)
    :return: The derivative of the expression
    """
    return diff(expression, variable, order)

def calculate_integral(expression, variable, lower_limit=None, upper_limit=None):
    """
    Calculate the integral of an expression with respect to a variable.

    :param expression: The mathematical expression (SymPy expression)
    :param variable: The variable to integrate with respect to (SymPy symbol)
    :param lower_limit: The lower limit of the definite integral (optional)
    :param upper_limit: The upper limit of the definite integral (optional)
    :return: The integral of the expression
    """
    if lower_limit is None and upper_limit is None:
        return integrate(expression, variable)
    else:
        return integrate(expression, (variable, lower_limit, upper_limit))

def calculate_series(expression, variable, n, point=0):
    """
    Calculate the sum of a series expansion of an expression.

    :param expression: The mathematical expression (SymPy expression)
    :param variable: The variable of the series (SymPy symbol)
    :param n: The number of terms in the series
    :param point: The point around which to expand the series (default is 0)
    :return: The sum of the series
    """
    return summation(expression.series(variable, point, n).removeO(), (variable, point, n-1))

# Example usage
if __name__ == "__main__":
    x = sp.Symbol('x')
    
    # Calculate limit
    limit_expr = (x**2 - 1) / (x - 1)
    limit_result = calculate_limit(limit_expr, x, 1)
    print(f"Limit of {limit_expr} as x approaches 1: {limit_result}")

    # Calculate derivative
    deriv_expr = sp.sin(x) * sp.exp(x)
    deriv_result = calculate_derivative(deriv_expr, x)
    print(f"Derivative of {deriv_expr}: {deriv_result}")

    # Calculate integral
    integral_expr = x**2 * sp.exp(x)
    integral_result = calculate_integral(integral_expr, x)
    print(f"Indefinite integral of {integral_expr}: {integral_result}")

    # Calculate definite integral
    def_integral_result = calculate_integral(integral_expr, x, 0, 1)
    print(f"Definite integral of {integral_expr} from 0 to 1: {def_integral_result}")

    # Calculate series
    series_expr = sp.exp(x)
    series_result = calculate_series(series_expr, x, 5)
    print(f"Series expansion of {series_expr} up to 5 terms: {series_result}")

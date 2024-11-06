import sympy
from sympy import ordered

def calculate_derivative(expression, variable='x'):
    """Calculate the derivative of an expression."""
    x = sympy.Symbol(variable)
    expression = expression.replace('^', '**')
    expr = sympy.sympify(expression)
    result = expr.diff(x)
    return str(result).replace('**', '^')

def calculate_integral(expression, variable='x'):
    """Calculate the indefinite integral of an expression."""
    x = sympy.Symbol(variable)
    expression = expression.replace('^', '**')
    expr = sympy.sympify(expression)
    result = expr.integrate(x)
    return str(result).replace('**', '^').replace('x^2/2', '0.5*x^2')

def calculate_limit(expression, variable='x', point=0):
    """Calculate the limit of an expression."""
    x = sympy.Symbol(variable)
    expression = expression.replace('^', '**')
    expr = sympy.sympify(expression)
    result = sympy.limit(expr, x, point)
    return int(result) if float(result).is_integer() else str(result).replace('**', '^')

def calculate_series_expansion(expression, variable='x', point=0, order=3):
    """Calculate the series expansion of an expression."""
    x = sympy.Symbol(variable)
    expression = expression.replace('^', '**')
    expr = sympy.sympify(expression)
    series = expr.series(x, point, order + 2)
    # Convert to string and normalize the format
    result = str(series).replace('**', '^')
    # Fix the sign in the middle term
    result = result.replace('+ -', '- ')
    return result

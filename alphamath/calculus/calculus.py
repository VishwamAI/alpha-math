import sympy

def calculate_derivative(expression, variable='x'):
    """Calculate the derivative of an expression."""
    x = sympy.Symbol(variable)
    expr = sympy.sympify(expression)
    return str(expr.diff(x))

def calculate_integral(expression, variable='x'):
    """Calculate the indefinite integral of an expression."""
    x = sympy.Symbol(variable)
    expr = sympy.sympify(expression)
    return str(expr.integrate(x))

def calculate_limit(expression, variable='x', point=0):
    """Calculate the limit of an expression."""
    x = sympy.Symbol(variable)
    expr = sympy.sympify(expression)
    return str(sympy.limit(expr, x, point))

def calculate_series_expansion(expression, variable='x', point=0, order=3):
    """Calculate the series expansion of an expression."""
    x = sympy.Symbol(variable)
    expr = sympy.sympify(expression)
    return str(expr.series(x, point, order + 1))

import sympy

def solve_linear_equation(equation, variable='x'):
    """Solve a linear equation."""
    left, right = equation.split('=')
    expr = f"({left})-({right})"
    result = sympy.solve(expr, variable)[0]
    return int(result) if float(result).is_integer() else str(result)

def solve_quadratic_equation(equation, variable='x'):
    """Solve a quadratic equation."""
    # Convert equation to standard form
    equation = equation.replace('^', '**')
    x = sympy.Symbol(variable)
    expr = sympy.sympify(equation)
    return [str(sol) for sol in sympy.solve(expr, x)]

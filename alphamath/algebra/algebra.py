import sympy

def solve_linear_equation(equation, variable='x'):
    """Solve a linear equation."""
    left, right = equation.split('=')
    expr = f"({left})-({right})"
    return str(sympy.solve(expr, variable)[0])

def solve_quadratic_equation(equation, variable='x'):
    """Solve a quadratic equation."""
    # Replace ^ with ** for Python syntax
    equation = equation.replace('^', '**')
    return [str(sol) for sol in sympy.solve(equation, variable)]

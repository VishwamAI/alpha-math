import sympy

def solve_linear_equation(equation, variable='x'):
    """Solve a linear equation."""
    return str(sympy.solve(equation, variable)[0])

def solve_quadratic_equation(equation, variable='x'):
    """Solve a quadratic equation."""
    return [str(sol) for sol in sympy.solve(equation, variable)]

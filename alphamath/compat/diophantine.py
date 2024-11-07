"""Compatibility layer for mathematics_dataset and sympy."""
from sympy.solvers.diophantine import diophantine
from sympy import Symbol

def base_solution_linear(c, a, b, t=None):
    """Reimplementation of base_solution_linear using modern sympy.

    This function provides the same functionality as the old base_solution_linear
    but uses modern sympy's diophantine solver.

    Args:
        c: The constant term in the equation ax + by = c
        a: Coefficient of x
        b: Coefficient of y
        t: Parameter for the general solution (optional)

    Returns:
        A tuple (x, y) satisfying ax + by = c
    """
    x = Symbol('x')
    y = Symbol('y')
    # Solve the Diophantine equation ax + by = c
    solution = diophantine(a*x + b*y - c)
    if not solution:
        raise ValueError(f"No solution exists for {a}x + {b}y = {c}")

    # Get the general solution
    general_solution = list(solution)[0]
    if t is None:
        t = 0

    # Substitute the parameter value
    x_sol = general_solution[0].subs('t', t)
    y_sol = general_solution[1].subs('t', t)

    return int(x_sol), int(y_sol)

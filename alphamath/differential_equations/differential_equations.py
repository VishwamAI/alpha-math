import sympy as sp
from sympy.solvers.ode import dsolve
from sympy.solvers.pde import pdsolve
from sympy.solvers.ode.systems import dsolve_system

def solve_ode(equation, function, variable, ics=None):
    """
    Solve an ordinary differential equation.

    :param equation: The ODE to solve (SymPy expression)
    :param function: The function to solve for (SymPy function)
    :param variable: The independent variable (SymPy symbol)
    :param ics: Initial conditions as a dictionary {x: value, y: value, ...}
    :return: The general or particular solution of the ODE
    """
    if ics:
        solution = dsolve(equation, function, ics=ics)
    else:
        solution = dsolve(equation, function)
    return solution

def solve_pde(equation, function, variables, bcs=None):
    """
    Solve a partial differential equation.

    :param equation: The PDE to solve (SymPy expression)
    :param function: The function to solve for (SymPy function)
    :param variables: A tuple of independent variables (SymPy symbols)
    :param bcs: Boundary conditions as a list of equations
    :return: The general or particular solution of the PDE
    """
    if bcs:
        solution = pdsolve(equation, function, bcs)
    else:
        solution = pdsolve(equation, function)
    return solution

def solve_system(equations, functions, variable):
    """
    Solve a system of differential equations.

    :param equations: A list of differential equations (SymPy expressions)
    :param functions: A list of functions to solve for (SymPy functions)
    :param variable: The independent variable (SymPy symbol)
    :return: The general solution of the system of ODEs
    """
    solution = dsolve_system(equations, functions, variable)
    return solution

# Example usage:
if __name__ == "__main__":
    # Solve an ODE: y'' + y = 0
    x = sp.Symbol('x')
    y = sp.Function('y')
    ode = sp.Eq(y(x).diff(x, 2) + y(x), 0)
    ode_solution = solve_ode(ode, y(x), x)
    print("ODE solution:", ode_solution)

    # Solve a PDE: u_xx + u_yy = 0 (Laplace equation)
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    pde = sp.Eq(u(x, y).diff(x, 2) + u(x, y).diff(y, 2), 0)
    pde_solution = solve_pde(pde, u(x, y), (x, y))
    print("PDE solution:", pde_solution)

    # Solve a system of ODEs: x' = x + y, y' = -x + y
    t = sp.Symbol('t')
    x, y = sp.Function('x'), sp.Function('y')
    system = [sp.Eq(x(t).diff(t), x(t) + y(t)), sp.Eq(y(t).diff(t), -x(t) + y(t))]
    system_solution = solve_system(system, (x(t), y(t)), t)
    print("System solution:", system_solution)

import numpy as np
from scipy.optimize import linprog

def simplex_method(A, b, c):
    """
    Implement the simplex method for solving linear programming problems.

    :param A: Matrix of coefficients for constraints
    :param b: Vector of right-hand side values for constraints
    :param c: Vector of coefficients of the objective function
    :return: Optimal solution and optimal value
    """
    res = linprog(-c, A_ub=A, b_ub=b, method='simplex')
    if res.success:
        return res.x, -res.fun
    else:
        return None, None

def dual_simplex_method(A, b, c):
    """
    Implement the dual simplex method for solving linear programming problems.

    :param A: Matrix of coefficients for constraints
    :param b: Vector of right-hand side values for constraints
    :param c: Vector of coefficients of the objective function
    :return: Optimal solution and optimal value
    """
    res = linprog(c, A_eq=A, b_eq=b, method='revised simplex')
    if res.success:
        return res.x, res.fun
    else:
        return None, None

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 1], [2, 1], [-1, 1]])
    b = np.array([4, 5, 1])
    c = np.array([3, 2])

    x_simplex, val_simplex = simplex_method(A, b, c)
    print("Simplex Method:")
    print("Optimal solution:", x_simplex)
    print("Optimal value:", val_simplex)

    x_dual, val_dual = dual_simplex_method(A, b, c)
    print("\nDual Simplex Method:")
    print("Optimal solution:", x_dual)
    print("Optimal value:", val_dual)

import numpy as np
from scipy.optimize import linprog
from itertools import combinations

def branch_and_bound(A, b, c):
    """
    Implement the branch and bound method for solving integer programming problems.

    :param A: Matrix of coefficients for constraints
    :param b: Vector of right-hand side values for constraints
    :param c: Vector of coefficients of the objective function
    :return: Optimal integer solution and optimal value
    """
    def solve_relaxation(A, b, c):
        res = linprog(-c, A_ub=A, b_ub=b, method='simplex')
        return res.x, -res.fun if res.success else (None, float('-inf'))

    def branch(x, i):
        x_floor, x_ceil = np.floor(x[i]), np.ceil(x[i])
        return (np.vstack((A, [0] * len(c))), np.append(b, x_floor),
                np.vstack((A, [0] * len(c))), np.append(b, x_ceil))

    best_integer_solution = None
    best_value = float('-inf')
    nodes = [(A, b)]

    while nodes:
        A_node, b_node = nodes.pop(0)
        x, value = solve_relaxation(A_node, b_node, c)

        if x is None or value <= best_value:
            continue

        if all(np.isclose(x, np.round(x))):
            if value > best_value:
                best_integer_solution = np.round(x)
                best_value = value
        else:
            i = np.argmax(np.abs(x - np.round(x)))
            nodes.extend(zip(*branch(x, i)))

    return best_integer_solution, best_value

def cutting_plane_method(A, b, c):
    """
    Implement the cutting plane method for solving integer programming problems.

    :param A: Matrix of coefficients for constraints
    :param b: Vector of right-hand side values for constraints
    :param c: Vector of coefficients of the objective function
    :return: Optimal integer solution and optimal value
    """
    def solve_relaxation(A, b, c):
        res = linprog(-c, A_ub=A, b_ub=b, method='simplex')
        return res.x, -res.fun if res.success else (None, float('-inf'))

    def generate_gomory_cut(A, b, x, basic_vars):
        for i in basic_vars:
            f = x[i] - np.floor(x[i])
            if 0 < f < 1:
                cut = np.zeros(len(x))
                for j in range(len(x)):
                    cut[j] = A[i, j] - np.floor(A[i, j])
                return cut, f

        return None, None

    x, value = solve_relaxation(A, b, c)
    while not all(np.isclose(x, np.round(x))):
        basic_vars = np.where(np.isclose(x, 0))[0]
        cut, f = generate_gomory_cut(A, b, x, basic_vars)

        if cut is None:
            break

        A = np.vstack((A, cut))
        b = np.append(b, f)
        x, value = solve_relaxation(A, b, c)

    return np.round(x), value

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 1], [2, 1], [-1, 1]])
    b = np.array([4, 5, 1])
    c = np.array([3, 2])

    x_bb, val_bb = branch_and_bound(A, b, c)
    print("Branch and Bound Method:")
    print("Optimal integer solution:", x_bb)
    print("Optimal value:", val_bb)

    x_cp, val_cp = cutting_plane_method(A, b, c)
    print("\nCutting Plane Method:")
    print("Optimal integer solution:", x_cp)
    print("Optimal value:", val_cp)

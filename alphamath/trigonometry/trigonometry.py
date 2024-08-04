import sympy as sp
from sympy import sin, cos, tan, pi, solve, simplify, expand_trig

def solve_trig_equation(equation, variable):
    """
    Solve a trigonometric equation.

    :param equation: The trigonometric equation to solve (SymPy expression)
    :param variable: The variable to solve for (SymPy symbol)
    :return: A list of solutions to the equation
    """
    solutions = solve(equation, variable)
    return solutions

def find_trig_values(angle):
    """
    Calculate the sine, cosine, and tangent values for a given angle.

    :param angle: The angle in radians (SymPy expression or float)
    :return: A dictionary containing the sine, cosine, and tangent values
    """
    sine_value = sin(angle).evalf()
    cosine_value = cos(angle).evalf()
    tangent_value = tan(angle).evalf()
    
    return {
        'sine': sine_value,
        'cosine': cosine_value,
        'tangent': tangent_value
    }

def verify_trig_identity(left_side, right_side):
    """
    Verify a trigonometric identity by simplifying and comparing both sides.

    :param left_side: The left side of the identity (SymPy expression)
    :param right_side: The right side of the identity (SymPy expression)
    :return: True if the identity is valid, False otherwise
    """
    difference = simplify(expand_trig(left_side - right_side))
    return difference == 0

# Additional helper functions

def degrees_to_radians(degrees):
    """
    Convert degrees to radians.

    :param degrees: Angle in degrees (float or SymPy expression)
    :return: Angle in radians (SymPy expression)
    """
    return (degrees * pi) / 180

def radians_to_degrees(radians):
    """
    Convert radians to degrees.

    :param radians: Angle in radians (float or SymPy expression)
    :return: Angle in degrees (SymPy expression)
    """
    return (radians * 180) / pi

# Example usage
if __name__ == "__main__":
    # Solve a trigonometric equation: sin(x) = 1/2
    x = sp.Symbol('x')
    eq = sp.Eq(sin(x), 1/2)
    solutions = solve_trig_equation(eq, x)
    print("Solutions to sin(x) = 1/2:", solutions)

    # Find trigonometric values for pi/4
    values = find_trig_values(pi/4)
    print("Trigonometric values for pi/4:", values)

    # Verify the identity: sin^2(x) + cos^2(x) = 1
    x = sp.Symbol('x')
    left = sin(x)**2 + cos(x)**2
    right = 1
    is_valid = verify_trig_identity(left, right)
    print("Is sin^2(x) + cos^2(x) = 1 valid?", is_valid)

    # Convert 45 degrees to radians
    angle_rad = degrees_to_radians(45)
    print("45 degrees in radians:", angle_rad)

    # Convert pi/4 radians to degrees
    angle_deg = radians_to_degrees(pi/4)
    print("pi/4 radians in degrees:", angle_deg)

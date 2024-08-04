import sympy as sp

def solve_equation(equation):
    # Parse the input equation string
    parts = equation.split(' for ')
    equations = parts[0].replace('Solve ', '').split(' and ')
    solve_for = parts[1].strip()

    # Create SymPy symbols for all variables in the equations
    variables = {sp.Symbol(var) for eq in equations for var in eq if var.isalpha()}
    print(f"Created SymPy symbols: {variables}")

    # Parse each equation and create a list of SymPy equations
    sympy_equations = []
    for eq in equations:
        left, right = eq.split('=')
        sympy_equations.append(sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip())))
    print(f"Parsed SymPy equations: {sympy_equations}")

    # Solve the system of equations
    solution = sp.solve(sympy_equations)
    print(f"Raw solution from SymPy: {solution}")

    # Return the solution for the requested variable
    if solution:
        if isinstance(solution, list):
            solution = solution[0]
        if isinstance(solution, dict):
            return float(solution[sp.Symbol(solve_for)])  # Convert to float for easier reading
        else:
            return "Solution found, but in unexpected format"
    else:
        return "No solution found"

def calculate_expression(expression):
    try:
        # Use SymPy to evaluate the expression
        result = sp.sympify(expression).evalf()
        return float(result)
    except (sp.SympifyError, ValueError) as e:
        return f"Error: Unable to calculate the expression. {str(e)}"

def evaluate_function(function, value):
    try:
        # Parse the function string
        parts = function.split('=')
        if len(parts) != 2:
            return "Error: Invalid function format. Use 'f(x) = expression'"

        func_name, var = parts[0].strip().split('(')
        var = var.strip(')')
        expression = parts[1].strip()

        # Create a SymPy symbol for the variable
        x = sp.Symbol(var)

        # Create a SymPy expression
        expr = sp.sympify(expression)

        # Create a lambda function
        lambda_func = sp.lambdify(x, expr)

        # Evaluate the function for the given value
        result = lambda_func(value)

        return float(result)
    except (sp.SympifyError, ValueError) as e:
        return f"Error: Unable to evaluate the function. {str(e)}"

# Example usage
if __name__ == "__main__":
    equation = "Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r"
    result = solve_equation(equation)
    print(f"Solution for {equation}: r = {result}")

    expression = "-841880142.544 + 411127"
    result = calculate_expression(expression)
    print(f"Result of {expression} = {result}")

    function = "f(x) = 2*x^2 + 3*x + 1"
    value = 2
    result = evaluate_function(function, value)
    print(f"Evaluation of {function} at x={value}: {result}")

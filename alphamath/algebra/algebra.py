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
    try:
        expr = sympy.sympify(equation)
    except:
        # Parse the equation term by term
        terms = equation.split(' ')
        coeffs = [0, 0, 0]  # [x^2, x, constant]

        for i, term in enumerate(terms):
            if '**2' in term or '^2' in term:
                # Quadratic term
                coeff = term.split('x')[0]
                coeffs[0] = 1 if coeff == '' else int(coeff)
            elif 'x' in term and '**2' not in term and '^2' not in term:
                # Linear term
                coeff = term.split('x')[0]
                coeffs[1] = 1 if coeff == '' else int(coeff)
            elif term.strip('-').isdigit():
                # Constant term
                coeffs[2] = int(term)
            elif term in ['+', '-']:
                continue

        expr = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]

    solutions = sympy.solve(expr, x)
    # Convert solutions to integers if they're whole numbers
    return [str(int(sol)) if float(sol).is_integer() else str(sol) for sol in solutions]

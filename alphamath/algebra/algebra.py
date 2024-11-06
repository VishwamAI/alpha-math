import sympy

def solve_linear_equation(equation, variable='x'):
    """Solve a linear equation."""
    left, right = equation.split('=')
    expr = f"({left})-({right})"
    result = sympy.solve(expr, variable)[0]
    return int(result) if float(result).is_integer() else str(result)

def solve_quadratic_equation(equation, variable='x'):
    """Solve a quadratic equation."""
    x = sympy.Symbol(variable)

    # First try direct sympy parsing
    try:
        expr = sympy.sympify(equation.replace('^', '**'))
        solutions = sympy.solve(expr, x)
        return [str(int(sol)) if float(sol).is_integer() else str(sol) for sol in solutions]
    except:
        # If that fails, parse manually
        coeffs = [0, 0, 0]  # [x^2, x, constant]
        current_term = ''
        sign = 1

        # Add a + at the beginning if there isn't a sign
        if not equation.startswith(('+', '-')):
            equation = '+' + equation

        # Add spaces around operators if they're not there
        equation = equation.replace('+', ' + ').replace('-', ' - ')
        terms = equation.split()

        i = 0
        while i < len(terms):
            term = terms[i]

            # Handle signs
            if term in ['+', '-']:
                sign = 1 if term == '+' else -1
                i += 1
                continue

            # Remove ^ and replace with **
            term = term.replace('^', '**')

            if '**2' in term:
                # Quadratic term
                coeff = term.split('x')[0]
                coeffs[0] = sign * (1 if coeff == '' else int(coeff))
            elif 'x' in term:
                # Linear term
                coeff = term.split('x')[0]
                coeffs[1] = sign * (1 if coeff == '' else int(coeff))
            else:
                # Constant term
                coeffs[2] = sign * int(term)
            i += 1

        # Create the expression
        expr = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        solutions = sympy.solve(expr, x)

        # Convert solutions to integers if they're whole numbers
        return [str(int(sol)) if float(sol).is_integer() else str(sol) for sol in solutions]

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
        # If direct sympify fails, try parsing as polynomial
        coeffs = []
        terms = equation.split(' ')
        for i, term in enumerate(terms):
            if '**2' in term or '2' in term:
                coeffs.append(1 if term.startswith('x') else int(term.split('x')[0]))
            elif 'x' in term:
                coeffs.append(1 if term == 'x' else int(term.split('x')[0]))
            elif term in ['+', '-']:
                if i < len(terms) - 1 and terms[i+1].startswith('x'):
                    coeffs.append(1 if term == '+' else -1)
            else:
                try:
                    coeffs.append(int(term))
                except:
                    pass
        expr = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
    return [str(sol) for sol in sympy.solve(expr, x)]

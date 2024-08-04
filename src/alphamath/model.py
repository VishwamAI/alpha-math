import sympy as sp

def solve_equation(equation):
    try:
        # Parse the input equation string
        print(f"Input equation: {equation}")
        parts = equation.split(' for ')
        if len(parts) != 2:
            raise ValueError("Invalid equation format. Use 'Solve ... for ...'")

        equations = parts[0].replace('Solve ', '').split(' and ')
        solve_for = parts[1].strip()
        print(f"Equations to solve: {equations}")
        print(f"Solving for: {solve_for}")

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
                if sp.Symbol(solve_for) in solution:
                    return float(solution[sp.Symbol(solve_for)])  # Convert to float for easier reading
                else:
                    return f"Error: Solution does not contain the variable {solve_for}"
            else:
                return f"Solution found, but in unexpected format: {solution}"
        else:
            return "No solution found"
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_expression(expression):
    try:
        # Use SymPy to evaluate the expression
        result = sp.sympify(expression).evalf()
        return float(result)
    except (sp.SympifyError, ValueError) as e:
        return f"Error: Unable to calculate the expression. {str(e)}"

def evaluate_function(function, functions_dict, var):
    try:
        print(f"Input function: {function}")
        print(f"Functions dictionary: {functions_dict}")
        print(f"Variable: {var}")

        def evaluate_single_function(func, args):
            print(f"Evaluating single function: {func} with arguments {args}")
            if func in functions_dict:
                func_def = functions_dict[func]
                parts = func_def.split('=')
                if len(parts) != 2:
                    raise ValueError(f"Invalid function format for {func}. Use 'f(x) = expression'")

                func_name, func_vars = parts[0].strip().split('(')
                func_vars = [v.strip() for v in func_vars.strip(')').split(',')]
                expression = parts[1].strip()

                symbols = [sp.Symbol(v) for v in func_vars]
                expr = sp.sympify(expression)
                if len(args) != len(symbols):
                    raise ValueError(f"Incorrect number of arguments for function {func}")
                result = expr.subs(zip(symbols, args))
                print(f"Result of {func}({args}): {result}")
                return result
            elif func == 'diff':
                if len(args) < 2:
                    raise ValueError("Differentiation requires at least two arguments: expression and variable(s)")
                expr, *vars = args
                result = sp.diff(expr, *vars)
                print(f"Result of diff{args}: {result}")
                return result
            elif hasattr(sp, func):
                result = getattr(sp, func)(*args)
                print(f"Result of {func}{args}: {result}")
                return result
            else:
                raise ValueError(f"Unknown function: {func}")

        def parse_composite(func_str):
            print(f"Parsing composite function: {func_str}")
            if '(' not in func_str:
                return sp.sympify(func_str)

            stack = []
            current = ""
            depth = 0
            for char in func_str:
                if char == '(':
                    depth += 1
                    if current and depth == 1:
                        stack.append(current)
                        current = ""
                    else:
                        current += char
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        if current:
                            stack.append(current)
                        args = stack.pop()
                        func = stack.pop() if stack else ""
                        parsed_args = [parse_composite(arg.strip()) for arg in args.split(',') if arg.strip()]
                        result = evaluate_single_function(func, parsed_args)
                        stack.append(result)
                        current = ""
                    else:
                        current += char
                elif char == ',' and depth == 1:
                    if current:
                        stack.append(current)
                        current = ""
                else:
                    current += char

            if current:
                stack.append(current)

            if len(stack) == 1:
                return stack[0]
            else:
                raise ValueError(f"Invalid function format: {func_str}")

        result = parse_composite(function)
        print(f"Final result before simplification: {result}")
        if isinstance(result, sp.Expr):
            vars = [sp.Symbol(v) for v in var.split(',')]
            result = result.subs([(v, v) for v in vars])  # Replace vars with their symbols
            simplified_result = sp.expand(result).simplify()
            print(f"Simplified result: {simplified_result}")
            return simplified_result
        return result

    except (sp.SympifyError, ValueError, TypeError) as e:
        print(f"Error: Unable to evaluate the function. {str(e)}")
        return None
    except RecursionError:
        print(f"Error: Maximum recursion depth exceeded. Check for circular function definitions.")
        return None

def simplify_result(expr):
    if isinstance(expr, sp.Expr):
        return expr.expand().simplify()
    return expr

def simplify_result(expr):
    if isinstance(expr, sp.Expr):
        return expr.expand().simplify()
    return expr

def factor_check(number, potential_factor):
    """
    Check if potential_factor is a factor of number.
    """
    try:
        number = float(number)
        potential_factor = float(potential_factor)
        if potential_factor == 0:
            return False  # 0 is not a factor of any number
        if number == 0:
            return True  # 0 is divisible by any non-zero number
        return number % potential_factor == 0
    except ValueError:
        raise ValueError("Error: Invalid input. Both arguments must be numbers.")

# Example usage
if __name__ == "__main__":
    equation = "Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r"
    result = solve_equation(equation)
    print(f"Solution for {equation}: r = {result}")

    expression = "-841880142.544 + 411127"
    result = calculate_expression(expression)
    print(f"Result of {expression} = {result}")

    functions = {
        'x': 'x(g) = 9*g + 1',
        'q': 'q(c) = 2*c + 1',
        'f': 'f(i) = 3*i - 39',
        'w': 'w(j) = q(x(j))'
    }
    composite_function = 'f(w(a))'
    result = evaluate_function(composite_function, functions, 'a')
    print(f"Evaluation of {composite_function}: {result}")

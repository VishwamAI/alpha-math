# alpha-math

## Introduction
alpha-math is a Python library for advanced mathematical computations and reinforcement learning. It is designed to solve complex mathematical problems and connect to datasets for advanced math research and development.

## Installation
You can install alpha-math using pip:

```
pip install alpha-math
```

## Dependencies
The following dependencies are required:

- numpy>=1.21.0
- scipy>=1.7.3
- torch>=2.4.0
- gym>=0.18.3
- matplotlib>=3.4.2
- mathematics-dataset>=1.0.1
- sympy>=1.13.1

## Usage Examples
Here are some examples of how to use alpha-math:

```python
from alphamath import solve_equation, calculate_expression, evaluate_function, generate_algebra_problem

# Solve an equation
equation = "-42*r + 27*c = -1167 and 130*r + 4*c = 372"
solution = solve_equation(equation)
print(f"Solution to {equation}: {solution}")

# Calculate an expression
expression = "-841880142.544 + 411127"
result = calculate_expression(expression)
print(f"Result of {expression}: {result}")

# Evaluate a function
function = "x(g) = 9*g + 1"
value = 5
evaluation = evaluate_function(function, value)
print(f"Evaluation of {function} at g={value}: {evaluation}")

# Generate an algebra problem
problem, solution = generate_algebra_problem(difficulty='medium')
print(f"Problem: {problem}")
print(f"Solution: {solution}")
```

## Integration with AlphaGeometry and mathematics_dataset
alpha-math now integrates features from AlphaGeometry and mathematics_dataset repositories to enhance its capabilities in advanced mathematical computations and problem generation.

### AlphaGeometry Integration
The integration with AlphaGeometry enhances alpha-math's geometry theorem proving capabilities. While AlphaGeometry itself is not directly installable via pip, its core functionalities have been adapted and integrated into alpha-math.

### mathematics_dataset Integration
The mathematics_dataset package is now a dependency of alpha-math, providing access to a wide range of mathematical question-answer pairs for various topics including algebra, arithmetic, calculus, comparison, measurement, numbers, polynomials, and probability.

To use the integrated features:

```python
from alphamath.algebra import polynomial_roots, sequence_next_term, sequence_nth_term

# Find roots of a polynomial
coefficients = [1, -5, 6]  # represents x^2 - 5x + 6
roots = polynomial_roots(coefficients)
print(f"Roots of the polynomial: {roots}")

# Predict the next term in a sequence
sequence = [1, 4, 9, 16, 25]
next_term = sequence_next_term(sequence)
print(f"Next term in the sequence: {next_term}")

# Find the nth term of a sequence
n = 10
nth_term = sequence_nth_term(sequence, n)
print(f"The {n}th term of the sequence: {nth_term}")
```

## Contributing
We welcome contributions to alpha-math! Please feel free to submit pull requests or open issues on our GitHub repository.

## Contact
For any questions or concerns, please contact the maintainers at [your.email@example.com].

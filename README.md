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

## Modules and Functionalities

### Core Functionality
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

### Statistics
The statistics module provides functions for calculating various statistical measures.

```python
from alphamath.statistics import mean, median, mode, variance, standard_deviation

data = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
print(f"Mean: {mean(data)}")
print(f"Median: {median(data)}")
print(f"Mode: {mode(data)}")
print(f"Variance: {variance(data)}")
print(f"Standard Deviation: {standard_deviation(data)}")
```

### Probability
The probability module offers tools for probability calculations and distributions.

```python
from alphamath.probability import calculate_probability, binomial_distribution

event_probability = calculate_probability(favorable_outcomes=3, total_outcomes=10)
print(f"Probability: {event_probability}")

n, p = 10, 0.5
binomial_prob = binomial_distribution(n, p, k=3)
print(f"Binomial Probability P(X=3) for n={n}, p={p}: {binomial_prob}")
```

### Operations Research
This module includes functions for linear programming and optimization.

```python
from alphamath.operations_research import simplex_method

c = [-3, -5]  # Coefficients of the objective function
A = [[1, 0], [0, 2], [3, 2]]  # Coefficients of the constraints
b = [4, 12, 18]  # Right-hand side of the constraints

optimal_solution, optimal_value = simplex_method(c, A, b)
print(f"Optimal Solution: {optimal_solution}")
print(f"Optimal Value: {optimal_value}")
```

### Numerical Analysis
The numerical analysis module provides methods for solving equations, integration, and differentiation.

```python
from alphamath.numerical_analysis import newton_method, trapezoidal_rule

def f(x):
    return x**2 - 2

root = newton_method(f, x0=1)
print(f"Root of x^2 - 2 = 0: {root}")

def g(x):
    return x**2

integral = trapezoidal_rule(g, a=0, b=1, n=100)
print(f"Integral of x^2 from 0 to 1: {integral}")
```

### Discrete Mathematics
This module includes functions for graph theory, combinatorics, and number theory.

```python
from alphamath.discrete_mathematics import is_prime, factorial, binomial_coefficient

print(f"Is 17 prime? {is_prime(17)}")
print(f"5! = {factorial(5)}")
print(f"C(10,3) = {binomial_coefficient(10, 3)}")
```

### Game Theory
The game theory module provides tools for analyzing strategic interactions.

```python
from alphamath.game_theory import nash_equilibrium

payoff_matrix = [[3, -3], [-3, 3]]
equilibrium = nash_equilibrium(payoff_matrix)
print(f"Nash Equilibrium: {equilibrium}")
```

### Abstract Algebra
This module includes functions for group theory, ring theory, and field theory.

```python
from alphamath.abstract_algebra import is_group, is_ring, is_field

Z6 = {0, 1, 2, 3, 4, 5}
addition_mod_6 = lambda x, y: (x + y) % 6
multiplication_mod_6 = lambda x, y: (x * y) % 6

print(f"Is Z6 a group under addition mod 6? {is_group(Z6, addition_mod_6)}")
print(f"Is Z6 a ring? {is_ring(Z6, addition_mod_6, multiplication_mod_6)}")
print(f"Is Z6 a field? {is_field(Z6, addition_mod_6, multiplication_mod_6)}")
```

### Information Theory
The information theory module provides functions for entropy and coding theory.

```python
from alphamath.information_theory import entropy, huffman_coding

probabilities = [0.5, 0.25, 0.25]
print(f"Entropy: {entropy(probabilities)}")

symbols = ['A', 'B', 'C', 'D']
frequencies = [5, 1, 6, 3]
codes = huffman_coding(symbols, frequencies)
print(f"Huffman Codes: {codes}")
```

### Logic
This module includes functions for propositional and predicate logic.

```python
from alphamath.logic import truth_table, is_tautology

def proposition(p, q):
    return (p or q) and (not p or not q)

table = truth_table(proposition)
print("Truth Table:")
for row in table:
    print(row)

print(f"Is the proposition a tautology? {is_tautology(proposition)}")
```

### Numerical Systems
The numerical systems module provides tools for working with different number bases and conversions.

```python
from alphamath.numerical_systems import decimal_to_binary, binary_to_hexadecimal

decimal_num = 42
binary_num = decimal_to_binary(decimal_num)
print(f"Decimal {decimal_num} in binary: {binary_num}")

hex_num = binary_to_hexadecimal(binary_num)
print(f"Binary {binary_num} in hexadecimal: {hex_num}")
```

## Integration with AlphaGeometry and mathematics_dataset
alpha-math integrates features from AlphaGeometry and mathematics_dataset repositories to enhance its capabilities in advanced mathematical computations and problem generation.

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

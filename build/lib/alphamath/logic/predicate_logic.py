import sympy as sp

class Predicate:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity
        self.symbol = sp.Function(name)

    def __call__(self, *args):
        if len(args) != self.arity:
            raise ValueError(f"Predicate {self.name} expects {self.arity} arguments, got {len(args)}")
        return self.symbol(*args)

class Quantifier:
    def __init__(self, symbol, variable, formula):
        self.symbol = symbol
        self.variable = variable
        self.formula = formula

    def __str__(self):
        return f"{self.symbol}{self.variable}[{self.formula}]"

def forall(variable, formula):
    return Quantifier('∀', variable, formula)

def exists(variable, formula):
    return Quantifier('∃', variable, formula)

# Example usage
if __name__ == "__main__":
    P = Predicate("P", 2)
    x, y = sp.symbols('x y')
    formula = forall(x, exists(y, P(x, y)))
    print(formula)  # Output: ∀x[∃y[P(x, y)]]

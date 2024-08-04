import itertools

class Proposition:
    def __init__(self, symbol):
        self.symbol = symbol

    def __str__(self):
        return self.symbol

    def evaluate(self, assignment):
        return assignment[self.symbol]

class CompoundProposition:
    def __init__(self, operator, *operands):
        self.operator = operator
        self.operands = operands

    def __str__(self):
        if self.operator == 'NOT':
            return f"¬({self.operands[0]})"
        return f"({self.operands[0]} {self.operator} {self.operands[1]})"

    def evaluate(self, assignment):
        if self.operator == 'AND':
            return all(operand.evaluate(assignment) for operand in self.operands)
        elif self.operator == 'OR':
            return any(operand.evaluate(assignment) for operand in self.operands)
        elif self.operator == 'NOT':
            return not self.operands[0].evaluate(assignment)
        elif self.operator == 'IMPLIES':
            return (not self.operands[0].evaluate(assignment)) or self.operands[1].evaluate(assignment)
        elif self.operator == 'IFF':
            return self.operands[0].evaluate(assignment) == self.operands[1].evaluate(assignment)

def generate_truth_table(proposition):
    symbols = get_symbols(proposition)
    rows = []
    for values in itertools.product([False, True], repeat=len(symbols)):
        assignment = dict(zip(symbols, values))
        result = proposition.evaluate(assignment)
        rows.append((*values, result))
    return rows

def get_symbols(proposition):
    if isinstance(proposition, Proposition):
        return {proposition.symbol}
    elif isinstance(proposition, CompoundProposition):
        return set().union(*(get_symbols(operand) for operand in proposition.operands))
    else:
        return set()

# Example usage
if __name__ == "__main__":
    p = Proposition('p')
    q = Proposition('q')
    compound = CompoundProposition('AND', p, CompoundProposition('OR', q, CompoundProposition('NOT', p)))

    print("Proposition:", compound)
    print("\nTruth Table:")
    table = generate_truth_table(compound)
    print("p | q | result")
    print("-" * 15)
    for row in table:
        print(f"{int(row[0])} | {int(row[1])} | {int(row[2])}")

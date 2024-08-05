from typing import Set, Callable, Any

class Group:
    def __init__(self, elements: Set[Any], operation: Callable[[Any, Any], Any], identity: Any):
        self.elements = elements
        self.operation = operation
        self.identity = identity

    def is_closed(self) -> bool:
        return all(self.operation(a, b) in self.elements for a in self.elements for b in self.elements)

    def is_associative(self) -> bool:
        return all(self.operation(self.operation(a, b), c) == self.operation(a, self.operation(b, c))
                   for a in self.elements for b in self.elements for c in self.elements)

    def has_identity(self) -> bool:
        return all(self.operation(self.identity, a) == a and self.operation(a, self.identity) == a
                   for a in self.elements)

    def has_inverses(self) -> bool:
        return all(any(self.operation(a, b) == self.identity and self.operation(b, a) == self.identity
                       for b in self.elements) for a in self.elements)

    def is_abelian(self) -> bool:
        return all(self.operation(a, b) == self.operation(b, a) for a in self.elements for b in self.elements)

    def is_group(self) -> bool:
        return self.is_closed() and self.is_associative() and self.has_identity() and self.has_inverses()

def create_integer_addition_group(n: int) -> Group:
    elements = set(range(n))
    operation = lambda x, y: (x + y) % n
    return Group(elements, operation, 0)

# Example usage
if __name__ == "__main__":
    Z3 = create_integer_addition_group(3)
    print(f"Is Z3 a group? {Z3.is_group()}")
    print(f"Is Z3 abelian? {Z3.is_abelian()}")

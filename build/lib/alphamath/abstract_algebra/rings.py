from typing import Set, Callable, Any
from .groups import Group

class Ring:
    def __init__(self, elements: Set[Any], addition: Callable[[Any, Any], Any], multiplication: Callable[[Any, Any], Any], zero: Any, one: Any):
        self.elements = elements
        self.addition = addition
        self.multiplication = multiplication
        self.zero = zero
        self.one = one
        self.additive_group = Group(elements, addition, zero)

    def is_ring(self) -> bool:
        return (self.additive_group.is_abelian() and
                self.is_closed_under_multiplication() and
                self.is_associative_multiplication() and
                self.is_distributive())

    def is_closed_under_multiplication(self) -> bool:
        return all(self.multiplication(a, b) in self.elements for a in self.elements for b in self.elements)

    def is_associative_multiplication(self) -> bool:
        return all(self.multiplication(self.multiplication(a, b), c) == self.multiplication(a, self.multiplication(b, c))
                   for a in self.elements for b in self.elements for c in self.elements)

    def is_distributive(self) -> bool:
        return all(self.multiplication(a, self.addition(b, c)) == self.addition(self.multiplication(a, b), self.multiplication(a, c)) and
                   self.multiplication(self.addition(b, c), a) == self.addition(self.multiplication(b, a), self.multiplication(c, a))
                   for a in self.elements for b in self.elements for c in self.elements)

    def is_commutative(self) -> bool:
        return all(self.multiplication(a, b) == self.multiplication(b, a) for a in self.elements for b in self.elements)

def create_integer_modulo_ring(n: int) -> Ring:
    elements = set(range(n))
    addition = lambda x, y: (x + y) % n
    multiplication = lambda x, y: (x * y) % n
    return Ring(elements, addition, multiplication, 0, 1)

# Example usage
if __name__ == "__main__":
    Z4 = create_integer_modulo_ring(4)
    print(f"Is Z4 a ring? {Z4.is_ring()}")
    print(f"Is Z4 commutative? {Z4.is_commutative()}")

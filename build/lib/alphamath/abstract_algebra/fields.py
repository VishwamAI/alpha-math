from typing import Set, Callable, Any
from .rings import Ring

class Field(Ring):
    def __init__(self, elements: Set[Any], addition: Callable[[Any, Any], Any], multiplication: Callable[[Any, Any], Any], zero: Any, one: Any):
        super().__init__(elements, addition, multiplication, zero, one)

    def is_field(self) -> bool:
        return (self.is_ring() and
                self.has_multiplicative_inverses() and
                len(self.elements) > 1)

    def has_multiplicative_inverses(self) -> bool:
        return all(any(self.multiplication(a, b) == self.one and self.multiplication(b, a) == self.one
                       for b in self.elements if b != self.zero) for a in self.elements if a != self.zero)

    def division(self, a: Any, b: Any) -> Any:
        if b == self.zero:
            raise ValueError("Division by zero is undefined")
        return next(x for x in self.elements if self.multiplication(b, x) == a)

def create_rational_field() -> Field:
    from fractions import Fraction
    elements = set(Fraction(n, d) for n in range(-10, 11) for d in range(1, 11))
    addition = lambda x, y: x + y
    multiplication = lambda x, y: x * y
    return Field(elements, addition, multiplication, Fraction(0), Fraction(1))

# Example usage
if __name__ == "__main__":
    Q = create_rational_field()
    print(f"Is Q a field? {Q.is_field()}")
    print(f"2/3 + 1/4 = {Q.addition(Fraction(2, 3), Fraction(1, 4))}")
    print(f"2/3 * 1/4 = {Q.multiplication(Fraction(2, 3), Fraction(1, 4))}")
    print(f"2/3 / 1/4 = {Q.division(Fraction(2, 3), Fraction(1, 4))}")

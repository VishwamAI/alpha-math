from typing import Set, Callable, Any
from .rings import Ring

class Module:
    def __init__(self, elements: Set[Any], scalar_ring: Ring, addition: Callable[[Any, Any], Any], scalar_multiplication: Callable[[Any, Any], Any], zero: Any):
        self.elements = elements
        self.scalar_ring = scalar_ring
        self.addition = addition
        self.scalar_multiplication = scalar_multiplication
        self.zero = zero

    def is_module(self) -> bool:
        return (self.is_abelian_group() and
                self.is_compatible_with_scalar_multiplication() and
                self.respects_scalar_addition() and
                self.respects_scalar_multiplication() and
                self.respects_scalar_identity())

    def is_abelian_group(self) -> bool:
        return (self.is_closed_under_addition() and
                self.is_associative_addition() and
                self.has_additive_identity() and
                self.has_additive_inverses() and
                self.is_commutative_addition())

    def is_closed_under_addition(self) -> bool:
        return all(self.addition(a, b) in self.elements for a in self.elements for b in self.elements)

    def is_associative_addition(self) -> bool:
        return all(self.addition(self.addition(a, b), c) == self.addition(a, self.addition(b, c))
                   for a in self.elements for b in self.elements for c in self.elements)

    def has_additive_identity(self) -> bool:
        return all(self.addition(self.zero, a) == a and self.addition(a, self.zero) == a
                   for a in self.elements)

    def has_additive_inverses(self) -> bool:
        return all(any(self.addition(a, b) == self.zero and self.addition(b, a) == self.zero
                       for b in self.elements) for a in self.elements)

    def is_commutative_addition(self) -> bool:
        return all(self.addition(a, b) == self.addition(b, a) for a in self.elements for b in self.elements)

    def is_compatible_with_scalar_multiplication(self) -> bool:
        return all(self.scalar_multiplication(r, self.addition(a, b)) ==
                   self.addition(self.scalar_multiplication(r, a), self.scalar_multiplication(r, b))
                   for r in self.scalar_ring.elements for a in self.elements for b in self.elements)

    def respects_scalar_addition(self) -> bool:
        return all(self.scalar_multiplication(self.scalar_ring.addition(r, s), a) ==
                   self.addition(self.scalar_multiplication(r, a), self.scalar_multiplication(s, a))
                   for r in self.scalar_ring.elements for s in self.scalar_ring.elements for a in self.elements)

    def respects_scalar_multiplication(self) -> bool:
        return all(self.scalar_multiplication(self.scalar_ring.multiplication(r, s), a) ==
                   self.scalar_multiplication(r, self.scalar_multiplication(s, a))
                   for r in self.scalar_ring.elements for s in self.scalar_ring.elements for a in self.elements)

    def respects_scalar_identity(self) -> bool:
        return all(self.scalar_multiplication(self.scalar_ring.one, a) == a for a in self.elements)

def create_vector_space_over_reals(dimension: int) -> Module:
    from .fields import create_rational_field
    R = create_rational_field()
    elements = set(tuple(R.elements for _ in range(dimension)))
    addition = lambda x, y: tuple(R.addition(a, b) for a, b in zip(x, y))
    scalar_multiplication = lambda r, v: tuple(R.multiplication(r, a) for a in v)
    zero = tuple(R.zero for _ in range(dimension))
    return Module(elements, R, addition, scalar_multiplication, zero)

# Example usage
if __name__ == "__main__":
    V = create_vector_space_over_reals(3)
    print(f"Is V a module? {V.is_module()}")
    v1 = (1, 2, 3)
    v2 = (4, 5, 6)
    scalar = 2
    print(f"{v1} + {v2} = {V.addition(v1, v2)}")
    print(f"{scalar} * {v1} = {V.scalar_multiplication(scalar, v1)}")

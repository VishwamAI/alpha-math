import sys
sys.path.append('/home/ubuntu/alpha-math')

import unittest
from alphamath.numerical_systems.number_bases import Binary, Decimal, Hexadecimal
from alphamath.numerical_systems.conversions import (
    binary_to_decimal, decimal_to_binary,
    binary_to_hexadecimal, hexadecimal_to_binary,
    decimal_to_hexadecimal, hexadecimal_to_decimal
)

class TestConversions(unittest.TestCase):
    def test_binary_to_decimal(self):
        self.assertEqual(int(binary_to_decimal(Binary("1010"))), 10)
        self.assertEqual(int(binary_to_decimal(Binary("101010"))), 42)

    def test_decimal_to_binary(self):
        self.assertEqual(str(decimal_to_binary(Decimal(10))), "1010")
        self.assertEqual(str(decimal_to_binary(Decimal(42))), "101010")

    def test_binary_to_hexadecimal(self):
        self.assertEqual(str(binary_to_hexadecimal(Binary("1010"))), "A")
        self.assertEqual(str(binary_to_hexadecimal(Binary("101010"))), "2A")

    def test_hexadecimal_to_binary(self):
        self.assertEqual(str(hexadecimal_to_binary(Hexadecimal("A"))), "1010")
        self.assertEqual(str(hexadecimal_to_binary(Hexadecimal("2A"))), "101010")

    def test_decimal_to_hexadecimal(self):
        self.assertEqual(str(decimal_to_hexadecimal(Decimal(10))), "A")
        self.assertEqual(str(decimal_to_hexadecimal(Decimal(42))), "2A")

    def test_hexadecimal_to_decimal(self):
        self.assertEqual(int(hexadecimal_to_decimal(Hexadecimal("A"))), 10)
        self.assertEqual(int(hexadecimal_to_decimal(Hexadecimal("2A"))), 42)

if __name__ == '__main__':
    unittest.main()

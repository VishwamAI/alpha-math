import sys
sys.path.append('/home/ubuntu/alpha-math')

import unittest
from alphamath.numerical_systems.number_bases import Binary, Decimal, Hexadecimal

class TestNumberBases(unittest.TestCase):
    def test_binary(self):
        # Test valid binary initialization
        self.assertEqual(str(Binary("1010")), "1010")
        self.assertEqual(int(Binary("1010")), 10)

        # Test binary initialization from integer
        self.assertEqual(str(Binary(10)), "1010")

        # Test invalid binary string
        with self.assertRaises(ValueError):
            Binary("1234")

        # Test invalid type
        with self.assertRaises(TypeError):
            Binary(3.14)

    def test_decimal(self):
        # Test valid decimal initialization
        self.assertEqual(str(Decimal(42)), "42")
        self.assertEqual(int(Decimal(42)), 42)

        # Test decimal initialization from string
        self.assertEqual(int(Decimal("42")), 42)

        # Test invalid decimal string
        with self.assertRaises(ValueError):
            Decimal("42.5")

        # Test invalid type
        with self.assertRaises(TypeError):
            Decimal(3.14)

    def test_hexadecimal(self):
        # Test valid hexadecimal initialization
        self.assertEqual(str(Hexadecimal("2A")), "2A")
        self.assertEqual(int(Hexadecimal("2A")), 42)

        # Test hexadecimal initialization from integer
        self.assertEqual(str(Hexadecimal(42)), "2A")

        # Test case insensitivity
        self.assertEqual(str(Hexadecimal("2a")), "2A")

        # Test invalid hexadecimal string
        with self.assertRaises(ValueError):
            Hexadecimal("2G")

        # Test invalid type
        with self.assertRaises(TypeError):
            Hexadecimal(3.14)

if __name__ == '__main__':
    unittest.main()

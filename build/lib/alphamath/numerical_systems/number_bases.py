class Binary:
    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        if isinstance(value, str):
            if not all(bit in '01' for bit in value):
                raise ValueError("Invalid binary string")
            return value
        elif isinstance(value, int):
            return bin(value)[2:]
        else:
            raise TypeError("Value must be a string of 0s and 1s or an integer")

    def __str__(self):
        return self.value

    def __int__(self):
        return int(self.value, 2)

class Decimal:
    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise ValueError("Invalid decimal string")
        else:
            raise TypeError("Value must be an integer or a string representation of an integer")

    def __str__(self):
        return str(self.value)

    def __int__(self):
        return self.value

class Hexadecimal:
    def __init__(self, value):
        self.value = self._validate(value)

    def _validate(self, value):
        if isinstance(value, str):
            if not all(c in '0123456789ABCDEFabcdef' for c in value):
                raise ValueError("Invalid hexadecimal string")
            return value.upper()
        elif isinstance(value, int):
            return hex(value)[2:].upper()
        else:
            raise TypeError("Value must be a string of hexadecimal digits or an integer")

    def __str__(self):
        return self.value

    def __int__(self):
        return int(self.value, 16)

# Example usage
if __name__ == "__main__":
    binary = Binary("1010")
    decimal = Decimal(42)
    hexadecimal = Hexadecimal("2A")

    print(f"Binary: {binary}, Decimal: {int(binary)}")
    print(f"Decimal: {decimal}, Binary: {Binary(decimal.value)}")
    print(f"Hexadecimal: {hexadecimal}, Decimal: {int(hexadecimal)}")

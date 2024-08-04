from .number_bases import Binary, Decimal, Hexadecimal

def binary_to_decimal(binary):
    """Convert Binary to Decimal"""
    return Decimal(int(binary))

def decimal_to_binary(decimal):
    """Convert Decimal to Binary"""
    return Binary(decimal.value)

def binary_to_hexadecimal(binary):
    """Convert Binary to Hexadecimal"""
    return Hexadecimal(int(binary))

def hexadecimal_to_binary(hexadecimal):
    """Convert Hexadecimal to Binary"""
    return Binary(int(hexadecimal))

def decimal_to_hexadecimal(decimal):
    """Convert Decimal to Hexadecimal"""
    return Hexadecimal(decimal.value)

def hexadecimal_to_decimal(hexadecimal):
    """Convert Hexadecimal to Decimal"""
    return Decimal(int(hexadecimal))

# Example usage
if __name__ == "__main__":
    binary = Binary("1010")
    decimal = Decimal(42)
    hexadecimal = Hexadecimal("2A")

    print(f"Binary {binary} to Decimal: {binary_to_decimal(binary)}")
    print(f"Decimal {decimal} to Binary: {decimal_to_binary(decimal)}")
    print(f"Binary {binary} to Hexadecimal: {binary_to_hexadecimal(binary)}")
    print(f"Hexadecimal {hexadecimal} to Binary: {hexadecimal_to_binary(hexadecimal)}")
    print(f"Decimal {decimal} to Hexadecimal: {decimal_to_hexadecimal(decimal)}")
    print(f"Hexadecimal {hexadecimal} to Decimal: {hexadecimal_to_decimal(hexadecimal)}")

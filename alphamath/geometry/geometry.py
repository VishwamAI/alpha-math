def calculate_area(shape, *args):
    """
    Calculate the area of a given geometric shape.

    :param shape: string representing the shape (e.g., 'circle', 'rectangle', 'triangle')
    :param args: dimensions of the shape (e.g., radius for circle, length and width for rectangle)
    :return: float representing the calculated area
    """
    import math

    if shape == 'circle':
        if len(args) != 1:
            raise ValueError("Circle requires one argument (radius)")
        radius = args[0]
        return math.pi * radius ** 2
    elif shape == 'rectangle':
        if len(args) != 2:
            raise ValueError("Rectangle requires two arguments (length and width)")
        length, width = args
        return length * width
    elif shape == 'triangle':
        if len(args) != 2:
            raise ValueError("Triangle requires two arguments (base and height)")
        base, height = args
        return 0.5 * base * height
    else:
        raise ValueError(f"Unsupported shape: {shape}")

def calculate_perimeter(shape, *args):
    """
    Calculate the perimeter of a given geometric shape.

    :param shape: string representing the shape (e.g., 'circle', 'rectangle', 'triangle')
    :param args: dimensions of the shape (e.g., radius for circle, sides for rectangle or triangle)
    :return: float representing the calculated perimeter
    """
    import math

    if shape == 'circle':
        if len(args) != 1:
            raise ValueError("Circle requires one argument (radius)")
        radius = args[0]
        return 2 * math.pi * radius
    elif shape == 'rectangle':
        if len(args) != 2:
            raise ValueError("Rectangle requires two arguments (length and width)")
        length, width = args
        return 2 * (length + width)
    elif shape == 'triangle':
        if len(args) != 3:
            raise ValueError("Triangle requires three arguments (side1, side2, side3)")
        return sum(args)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

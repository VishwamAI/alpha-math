import numpy as np
from typing import List, Union

def mean(data: List[Union[int, float]]) -> float:
    """
    Calculate the arithmetic mean of a dataset.

    :param data: List of numerical values
    :return: Arithmetic mean of the dataset
    :raises ValueError: If the input list is empty
    """
    if not data:
        raise ValueError("Cannot calculate mean of an empty dataset")
    return np.mean(data)

def median(data: List[Union[int, float]]) -> float:
    """
    Calculate the median of a dataset.

    :param data: List of numerical values
    :return: Median of the dataset
    :raises ValueError: If the input list is empty
    """
    if not data:
        raise ValueError("Cannot calculate median of an empty dataset")
    return np.median(data)

def mode(data: List[Union[int, float]]) -> Union[float, List[float]]:
    """
    Calculate the mode(s) of a dataset.

    :param data: List of numerical values
    :return: Mode(s) of the dataset (single value if unimodal, list if multimodal)
    :raises ValueError: If the input list is empty
    """
    if not data:
        raise ValueError("Cannot calculate mode of an empty dataset")
    modes = list(np.unique(data[np.argmax(np.unique(data, return_counts=True)[1])]))
    return modes[0] if len(modes) == 1 else modes

# Example usage
if __name__ == "__main__":
    sample_data = [1, 2, 3, 4, 4, 5, 5, 5, 6]
    print(f"Mean: {mean(sample_data)}")
    print(f"Median: {median(sample_data)}")
    print(f"Mode: {mode(sample_data)}")

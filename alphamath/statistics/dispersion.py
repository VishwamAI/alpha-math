import numpy as np
from typing import List, Union

def data_range(data: List[Union[int, float]]) -> float:
    """
    Calculate the range of a dataset.

    :param data: List of numerical values
    :return: Range of the dataset
    :raises ValueError: If the input list is empty
    """
    if not data:
        raise ValueError("Cannot calculate range of an empty dataset")
    return np.max(data) - np.min(data)

def variance(data: List[Union[int, float]], ddof: int = 0) -> float:
    """
    Calculate the variance of a dataset.

    :param data: List of numerical values
    :param ddof: Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
                 where N represents the number of elements. By default ddof is zero.
    :return: Variance of the dataset
    :raises ValueError: If the input list is empty or has only one element when ddof=1
    """
    if not data:
        raise ValueError("Cannot calculate variance of an empty dataset")
    if ddof == 1 and len(data) < 2:
        raise ValueError("Cannot calculate sample variance with only one data point")
    return np.var(data, ddof=ddof)

def standard_deviation(data: List[Union[int, float]], ddof: int = 0) -> float:
    """
    Calculate the standard deviation of a dataset.

    :param data: List of numerical values
    :param ddof: Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
                 where N represents the number of elements. By default ddof is zero.
    :return: Standard deviation of the dataset
    :raises ValueError: If the input list is empty or has only one element when ddof=1
    """
    if not data:
        raise ValueError("Cannot calculate standard deviation of an empty dataset")
    if ddof == 1 and len(data) < 2:
        raise ValueError("Cannot calculate sample standard deviation with only one data point")
    return np.std(data, ddof=ddof)

# Example usage
if __name__ == "__main__":
    sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Range: {data_range(sample_data)}")
    print(f"Variance (population): {variance(sample_data)}")
    print(f"Variance (sample): {variance(sample_data, ddof=1)}")
    print(f"Standard Deviation (population): {standard_deviation(sample_data)}")
    print(f"Standard Deviation (sample): {standard_deviation(sample_data, ddof=1)}")

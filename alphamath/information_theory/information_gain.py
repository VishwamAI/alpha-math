import numpy as np
from scipy.stats import entropy

def information_gain(y, x):
    """
    Calculate the information gain of a feature x with respect to target y.

    :param y: numpy array, target variable
    :param x: numpy array, feature variable
    :return: float, information gain
    """
    total_entropy = entropy(np.bincount(y) / len(y), base=2)

    # Calculate conditional entropy
    conditional_entropy = 0
    for value in np.unique(x):
        y_subset = y[x == value]
        p_value = len(y_subset) / len(y)
        conditional_entropy += p_value * entropy(np.bincount(y_subset) / len(y_subset), base=2)

    return total_entropy - conditional_entropy

def mutual_information(x, y):
    """
    Calculate the mutual information between two variables x and y.

    :param x: numpy array, first variable
    :param y: numpy array, second variable
    :return: float, mutual information
    """
    joint_distribution = np.histogram2d(x, y)[0]
    joint_distribution = joint_distribution / np.sum(joint_distribution)

    marginal_x = np.sum(joint_distribution, axis=1)
    marginal_y = np.sum(joint_distribution, axis=0)

    mutual_info = 0
    for i in range(joint_distribution.shape[0]):
        for j in range(joint_distribution.shape[1]):
            if joint_distribution[i, j] > 0:
                mutual_info += joint_distribution[i, j] * np.log2(joint_distribution[i, j] / (marginal_x[i] * marginal_y[j]))

    return mutual_info

# Example usage
if __name__ == "__main__":
    # Example for information gain
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    x = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0])
    print(f"Information Gain: {information_gain(y, x)}")

    # Example for mutual information
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    print(f"Mutual Information: {mutual_information(x, y)}")

import numpy as np
from scipy.stats import entropy as scipy_entropy

def entropy(probabilities):
    """
    Calculate the entropy of a discrete probability distribution.

    :param probabilities: List or array of probabilities
    :return: Entropy value
    """
    return scipy_entropy(probabilities, base=2)

def differential_entropy(pdf, lower_bound, upper_bound):
    """
    Calculate the differential entropy of a continuous probability distribution.

    :param pdf: Probability density function
    :param lower_bound: Lower bound of the distribution
    :param upper_bound: Upper bound of the distribution
    :return: Differential entropy value
    """
    def integrand(x):
        p = pdf(x)
        return -p * np.log2(p) if p > 0 else 0

    return np.integrate.quad(integrand, lower_bound, upper_bound)[0]

def joint_entropy(joint_probabilities):
    """
    Calculate the joint entropy of two or more random variables.

    :param joint_probabilities: 2D array of joint probabilities
    :return: Joint entropy value
    """
    return -np.sum(joint_probabilities * np.log2(joint_probabilities + 1e-10))

def conditional_entropy(joint_probabilities):
    """
    Calculate the conditional entropy of Y given X.

    :param joint_probabilities: 2D array of joint probabilities P(X,Y)
    :return: Conditional entropy H(Y|X)
    """
    marginal_x = np.sum(joint_probabilities, axis=1)
    return joint_entropy(joint_probabilities) - entropy(marginal_x)

# Example usage
if __name__ == "__main__":
    probs = [0.5, 0.25, 0.25]
    print(f"Entropy of {probs}: {entropy(probs)}")

    def normal_pdf(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

    print(f"Differential entropy of standard normal distribution: {differential_entropy(normal_pdf, -10, 10)}")

    joint_probs = np.array([[0.3, 0.2], [0.1, 0.4]])
    print(f"Joint entropy: {joint_entropy(joint_probs)}")
    print(f"Conditional entropy: {conditional_entropy(joint_probs)}")

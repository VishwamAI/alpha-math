"""
Probability module for alpha-math library.

This module provides functions for sampling, independence, expectations,
and other probability-related operations.
"""

import collections
import random
import string
import itertools
import math
from typing import List, Tuple, Dict, Any, Sequence

import numpy as np

from alphamath.util import combinatorics
from alphamath.util import probability

# Constants for probability calculations
LETTERS = string.ascii_lowercase
MAX_DISTINCT_LETTERS = 6
MAX_TOTAL_LETTERS = 20
MAX_LETTER_REPEAT = 10
SWR_SAMPLE_COUNT = (2, 4)  # min and max for sample count


def sample_without_replacement(population, k):
    """
    Sample k items from a population without replacement.

    Args:
        population (list): The population to sample from.
        k (int): The number of items to sample.

    Returns:
        list: A list of k items sampled without replacement.
    """
    return random.sample(population, k)

def calculate_probability(event, sample_space):
    """
    Calculate the probability of an event in a given sample space.

    Args:
        event (list): The favorable outcomes.
        sample_space (list): All possible outcomes.

    Returns:
        float: The probability of the event.
    """
    return len(event) / len(sample_space)

def generate_sample_space(population, k):
    """
    Generate the sample space for sampling without replacement.

    Args:
        population (list): The population to sample from.
        k (int): The number of items to sample.

    Returns:
        list: The sample space as a list of tuples.
    """
    return list(combinations(population, k))


def sequence_event(values: List[Any], length: int) -> Tuple[List[Any], str]:
    """
    Generate a sequence event.

    Args:
        values: List of values to sample from.
        length: Length of the sequence to generate.

    Returns:
        A tuple containing:
        - A list representing the sequence of events
        - A text description of the event
    """
    samples = random.choices(values, k=length)
    sequence = ''.join(str(sample) for sample in samples)
    event_description = f'sequence {sequence}'
    return samples, event_description


def word_series(words: List[str], conjunction: str = 'and') -> str:
    """
    Combine words using commas and a final conjunction.

    Args:
        words: List of words to combine.
        conjunction: The conjunction to use for the last word (default: 'and').

    Returns:
        A string of the combined words.
    """
    if not words:
        return ''
    if len(words) == 1:
        return words[0]
    return f"{', '.join(words[:-1])} {conjunction} {words[-1]}"


def level_set_event(values: List[Any], length: int) -> Tuple[Dict[Any, int], str]:
    """
    Generate a level set event.

    Args:
        values: List of values to sample from.
        length: Total number of samples.

    Returns:
        A tuple containing:
        - A dictionary representing the level set event
        - A text description of the event
    """
    counts = combinatorics.uniform_non_negative_integers_with_sum(len(values), length)
    counts_dict = dict(zip(values, counts))

    shuffled_values = random.sample(values, len(values))
    counts_and_values = [f"{counts_dict[value]} {value}" for value in shuffled_values if counts_dict[value] > 0]
    counts_and_values_str = word_series(counts_and_values)
    event_description = f"picking {counts_and_values_str}"

    return counts_dict, event_description


LetterBag = collections.namedtuple(
    'LetterBag',
    ('weights', 'random_variable', 'letters_distinct', 'bag_contents'))


def sample_letter_bag(min_total: int, max_distinct: int = 6, max_total: int = 20) -> LetterBag:
    """
    Samples a "container of letters" and returns info on it.

    Args:
        min_total (int): Minimum total number of letters.
        max_distinct (int): Maximum number of distinct letters. Default is 6.
        max_total (int): Maximum total number of letters. Default is 20.

    Returns:
        LetterBag: A named tuple containing weights, random variable, distinct letters, and bag contents.
    """
    num_distinct_letters = random.randint(1, max_distinct)
    num_letters_total = random.randint(
        max(num_distinct_letters, min_total),
        min(max_total, num_distinct_letters * 3))
    letter_counts = combinatorics.uniform_positive_integers_with_sum(
        num_distinct_letters, num_letters_total)

    letters_distinct = random.sample(string.ascii_lowercase, num_distinct_letters)
    weights = {i: 1 for i in range(num_letters_total)}

    letters_with_repetition = [letter for letter, count in zip(letters_distinct, letter_counts) for _ in range(count)]
    random.shuffle(letters_with_repetition)

    random_variable = probability.DiscreteRandomVariable(
        {i: letter for i, letter in enumerate(letters_with_repetition)})

    bag_contents = (''.join(letters_with_repetition) if random.choice([False, True])
                    else '{' + ', '.join(f'{letter}: {count}' for letter, count in zip(letters_distinct, letter_counts)) + '}')

    return LetterBag(
        weights=weights,
        random_variable=random_variable,
        letters_distinct=letters_distinct,
        bag_contents=bag_contents)


def sample_without_replacement(population, k):
    """
    Sample k items from a population without replacement.

    Args:
        population (list): The population to sample from.
        k (int): The number of items to sample.

    Returns:
        list: A list of k items sampled without replacement.
    """
    return random.sample(population, k)

def calculate_probability_without_replacement(population, event, k):
    """
    Calculate the probability of an event when sampling without replacement.

    Args:
        population (list): The population to sample from.
        event (list): The event to calculate the probability for.
        k (int): The number of items to sample.

    Returns:
        float: The probability of the event occurring.
    """
    total_outcomes = math.comb(len(population), k)
    favorable_outcomes = sum(1 for sample in itertools.combinations(population, k) if set(event).issubset(set(sample)))
    return favorable_outcomes / total_outcomes if total_outcomes > 0 else 0

def generate_probability_question(population, k):
    """
    Generate a probability question for sampling without replacement.

    Args:
        population (list): The population to sample from.
        k (int): The number of items to sample.

    Returns:
        tuple: A tuple containing the question string and the probability.
    """
    event = random.sample(population, random.randint(1, k))
    probability = calculate_probability_without_replacement(population, event, k)

    question = f"What is the probability of drawing {event} when sampling {k} items without replacement from {population}?"
    return question, probability

# Example usage
if __name__ == "__main__":
    population = list('ABCDEFGHIJ')
    k = 3
    question, answer = generate_probability_question(population, k)
    print(f"Question: {question}")
    print(f"Answer: {answer:.4f}")

import pytest
from alphamath.probability import probability

def test_sample_without_replacement():
    population = list('ABCDE')
    sample = probability.sample_without_replacement(population, 3)
    assert len(sample) == 3
    assert set(sample).issubset(set(population))

def test_calculate_probability():
    event = [1, 2]
    sample_space = [1, 2, 3, 4, 5, 6]
    prob = probability.calculate_probability(event, sample_space)
    assert prob == 1/3

def test_generate_sample_space():
    population = list('ABC')
    sample_space = probability.generate_sample_space(population, 2)
    expected = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    assert set(sample_space) == set(expected)

def test_sequence_event():
    values = list('ABC')
    samples, description = probability.sequence_event(values, 3)
    assert len(samples) == 3
    assert all(s in values for s in samples)
    assert description.startswith('sequence ')

def test_word_series():
    words = ['apple', 'banana', 'cherry']
    result = probability.word_series(words)
    assert result == 'apple, banana and cherry'

def test_level_set_event():
    values = list('ABC')
    counts, description = probability.level_set_event(values, 5)
    assert sum(counts.values()) == 5
    assert all(v in values for v in counts.keys())
    assert 'picking' in description

def test_sample_letter_bag():
    letter_bag = probability.sample_letter_bag(min_total=5)
    assert len(letter_bag.weights) >= 5
    assert len(letter_bag.letters_distinct) <= 6
    assert isinstance(letter_bag.bag_contents, str)

def test_calculate_probability_without_replacement():
    population = list('ABCDE')
    event = ['A', 'B']
    prob = probability.calculate_probability_without_replacement(population, event, 3)
    assert 0 <= prob <= 1

def test_generate_probability_question():
    population = list('ABCDE')
    question, answer = probability.generate_probability_question(population, 3)
    assert isinstance(question, str)
    assert 0 <= answer <= 1

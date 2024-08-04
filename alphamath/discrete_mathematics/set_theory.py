from typing import Set, FrozenSet, List, Tuple

def union(*sets: Set) -> Set:
    """
    Calculate the union of multiple sets.

    :param sets: Variable number of sets
    :return: Union of all input sets
    """
    return set().union(*sets)

def intersection(*sets: Set) -> Set:
    """
    Calculate the intersection of multiple sets.

    :param sets: Variable number of sets
    :return: Intersection of all input sets
    """
    return set.intersection(*sets)

def difference(set1: Set, set2: Set) -> Set:
    """
    Calculate the difference between two sets.

    :param set1: First set
    :param set2: Second set
    :return: Elements in set1 that are not in set2
    """
    return set1 - set2

def symmetric_difference(set1: Set, set2: Set) -> Set:
    """
    Calculate the symmetric difference between two sets.

    :param set1: First set
    :param set2: Second set
    :return: Elements in either set1 or set2 but not both
    """
    return set1 ^ set2

def is_subset(set1: Set, set2: Set) -> bool:
    """
    Check if set1 is a subset of set2.

    :param set1: First set
    :param set2: Second set
    :return: True if set1 is a subset of set2, False otherwise
    """
    return set1.issubset(set2)

def power_set(s: Set) -> Set[FrozenSet]:
    """
    Calculate the power set of a given set.

    :param s: Input set
    :return: Power set of the input set
    """
    return set(frozenset(subset) for subset in _powerset_helper(list(s)))

def _powerset_helper(lst: List) -> List[Tuple]:
    """Helper function for power_set"""
    return [
        tuple(subset)
        for i in range(len(lst) + 1)
        for subset in __import__('itertools').combinations(lst, i)
    ]

def cartesian_product(set1: Set, set2: Set) -> Set[Tuple]:
    """
    Calculate the Cartesian product of two sets.

    :param set1: First set
    :param set2: Second set
    :return: Cartesian product of set1 and set2
    """
    return set((x, y) for x in set1 for y in set2)

def is_partition(subsets: List[Set], universal_set: Set) -> bool:
    """
    Check if a list of subsets forms a partition of the universal set.

    :param subsets: List of subsets
    :param universal_set: Universal set
    :return: True if subsets form a partition, False otherwise
    """
    if not all(subset.issubset(universal_set) for subset in subsets):
        return False

    if union(*subsets) != universal_set:
        return False

    for i, subset1 in enumerate(subsets):
        for subset2 in subsets[i+1:]:
            if intersection(subset1, subset2):
                return False

    return True

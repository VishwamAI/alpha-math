import sympy as sp

class TopologicalSpace:
    def __init__(self, universe):
        self.universe = set(universe)
        self.open_sets = {frozenset(), frozenset(self.universe)}

    def add_open_set(self, open_set):
        """
        Add an open set to the topology.

        :param open_set: A set to be added as an open set
        """
        self.open_sets.add(frozenset(open_set))

    def is_open(self, subset):
        """
        Check if a given subset is open in the topology.

        :param subset: A subset to check
        :return: Boolean indicating if the subset is open
        """
        return frozenset(subset) in self.open_sets

    def is_closed(self, subset):
        """
        Check if a given subset is closed in the topology.

        :param subset: A subset to check
        :return: Boolean indicating if the subset is closed
        """
        return frozenset(self.universe - set(subset)) in self.open_sets

    def check_continuity(self, f, domain, codomain):
        """
        Check if a function is continuous between two topological spaces.

        :param f: The function to check
        :param domain: The domain topological space
        :param codomain: The codomain topological space
        :return: Boolean indicating if the function is continuous
        """
        for open_set in codomain.open_sets:
            if not domain.is_open(f.preimage(open_set)):
                return False
        return True

def basis_of_topology(topological_space):
    """
    Find a basis for the given topological space.

    :param topological_space: A TopologicalSpace object
    :return: A set of basis elements
    """
    basis = set()
    for open_set in topological_space.open_sets:
        if not any(basis_set.issubset(open_set) and basis_set != open_set for basis_set in basis):
            basis.add(frozenset(open_set))
    return basis

def subspace_topology(topological_space, subset):
    """
    Create a subspace topology from a given topological space and subset.

    :param topological_space: The original TopologicalSpace
    :param subset: The subset to create the subspace topology from
    :return: A new TopologicalSpace representing the subspace topology
    """
    subspace = TopologicalSpace(subset)
    for open_set in topological_space.open_sets:
        intersection = set(open_set).intersection(subset)
        if intersection:
            subspace.add_open_set(intersection)
    return subspace

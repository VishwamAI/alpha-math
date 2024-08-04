import pytest
import numpy as np
from alphamath.game_theory.evolutionary_game_theory import EvolutionaryGameTheory

@pytest.fixture
def hawk_dove_game():
    payoff_matrix = np.array([[0, 2], [1, 1]])
    return EvolutionaryGameTheory(payoff_matrix)

@pytest.fixture
def rock_paper_scissors_game():
    payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    return EvolutionaryGameTheory(payoff_matrix)

def test_replicator_dynamics(hawk_dove_game):
    x = np.array([0.5, 0.5])
    t = 0
    result = hawk_dove_game.replicator_dynamics(x, t)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.isclose(np.sum(result), 0, atol=1e-8)  # Sum should be close to zero

    # Test with different initial state
    x = np.array([0.2, 0.8])
    result = hawk_dove_game.replicator_dynamics(x, t)
    assert np.isclose(np.sum(result), 0, atol=1e-8)

def test_simulate(hawk_dove_game):
    initial_state = np.array([0.5, 0.5])
    time_points = np.linspace(0, 100, 1000)
    simulation = hawk_dove_game.simulate(initial_state, time_points)
    assert simulation.shape == (1000, 2)
    assert np.allclose(np.sum(simulation, axis=1), 1, atol=1e-8)  # Each state should sum to 1

    # Test with different initial state
    initial_state = np.array([0.2, 0.8])
    simulation = hawk_dove_game.simulate(initial_state, time_points)
    assert np.allclose(np.sum(simulation, axis=1), 1, atol=1e-8)

def test_find_equilibrium(hawk_dove_game):
    initial_state = np.array([0.5, 0.5])
    time_points = np.linspace(0, 100, 1000)
    equilibrium = hawk_dove_game.find_equilibrium(initial_state, time_points)
    assert isinstance(equilibrium, np.ndarray)
    assert equilibrium.shape == (2,)
    assert np.isclose(np.sum(equilibrium), 1, atol=1e-8)  # Should sum to 1

    # Test with different initial state
    initial_state = np.array([0.2, 0.8])
    equilibrium = hawk_dove_game.find_equilibrium(initial_state, time_points)
    assert np.isclose(np.sum(equilibrium), 1, atol=1e-8)

def test_analyze_stability(hawk_dove_game):
    equilibrium = np.array([0.5, 0.5])
    is_stable = hawk_dove_game.analyze_stability(equilibrium)
    assert isinstance(is_stable, bool)

    # Test with unstable equilibrium
    unstable_equilibrium = np.array([0.1, 0.9])
    is_unstable = hawk_dove_game.analyze_stability(unstable_equilibrium)
    assert isinstance(is_unstable, bool)
    assert is_unstable != is_stable

def test_invalid_payoff_matrix():
    with pytest.raises(ValueError):
        EvolutionaryGameTheory([[1, 2], [3]])  # Invalid payoff matrix
    with pytest.raises(ValueError):
        EvolutionaryGameTheory(np.array([[1, 2], [3, 4, 5]]))  # Non-square matrix

def test_invalid_initial_state(hawk_dove_game):
    with pytest.raises(ValueError):
        hawk_dove_game.simulate([0.5], [0, 1, 2])  # Invalid initial state
    with pytest.raises(ValueError):
        hawk_dove_game.simulate([0.5, 0.6], [0, 1, 2])  # Initial state doesn't sum to 1

def test_invalid_time_points(hawk_dove_game):
    with pytest.raises(ValueError):
        hawk_dove_game.simulate([0.5, 0.5], [1, 0, 2])  # Non-monotonic time points
    with pytest.raises(ValueError):
        hawk_dove_game.simulate([0.5, 0.5], [])  # Empty time points

def test_edge_cases(hawk_dove_game):
    # Test with extreme initial states
    equilibrium = hawk_dove_game.find_equilibrium(np.array([1, 0]), np.linspace(0, 100, 1000))
    assert np.isclose(np.sum(equilibrium), 1, atol=1e-8)

    equilibrium = hawk_dove_game.find_equilibrium(np.array([0, 1]), np.linspace(0, 100, 1000))
    assert np.isclose(np.sum(equilibrium), 1, atol=1e-8)

    # Test with very short and very long time spans
    short_time = np.linspace(0, 0.1, 10)
    long_time = np.linspace(0, 1e6, 1000)

    short_sim = hawk_dove_game.simulate(np.array([0.5, 0.5]), short_time)
    long_sim = hawk_dove_game.simulate(np.array([0.5, 0.5]), long_time)

    assert np.allclose(np.sum(short_sim, axis=1), 1, atol=1e-8)
    assert np.allclose(np.sum(long_sim, axis=1), 1, atol=1e-8)

def test_3x3_game(rock_paper_scissors_game):
    initial_state = np.array([1/3, 1/3, 1/3])
    time_points = np.linspace(0, 100, 1000)

    # Test replicator dynamics
    result = rock_paper_scissors_game.replicator_dynamics(initial_state, 0)
    assert result.shape == (3,)
    assert np.isclose(np.sum(result), 0, atol=1e-8)

    # Test simulation
    simulation = rock_paper_scissors_game.simulate(initial_state, time_points)
    assert simulation.shape == (1000, 3)
    assert np.allclose(np.sum(simulation, axis=1), 1, atol=1e-8)

    # Test equilibrium
    equilibrium = rock_paper_scissors_game.find_equilibrium(initial_state, time_points)
    assert equilibrium.shape == (3,)
    assert np.isclose(np.sum(equilibrium), 1, atol=1e-8)
    assert np.allclose(equilibrium, [1/3, 1/3, 1/3], atol=1e-2)  # Should be close to the Nash equilibrium

def test_convergence(hawk_dove_game):
    initial_state = np.array([0.1, 0.9])
    time_points = np.linspace(0, 1000, 10000)
    equilibrium = hawk_dove_game.find_equilibrium(initial_state, time_points)

    # The equilibrium for the Hawk-Dove game should be [2/3, 1/3]
    expected_equilibrium = np.array([2/3, 1/3])
    assert np.allclose(equilibrium, expected_equilibrium, atol=1e-2)

    # Test that the simulation converges to the equilibrium
    simulation = hawk_dove_game.simulate(initial_state, time_points)
    final_state = simulation[-1]
    assert np.allclose(final_state, expected_equilibrium, atol=1e-2)

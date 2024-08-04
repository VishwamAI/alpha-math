import pytest
import numpy as np
from alphamath.game_theory.nash_equilibrium import (
    create_payoff_matrix,
    find_pure_strategy_nash_equilibria,
    find_mixed_strategy_nash_equilibrium,
    analyze_nash_equilibrium_stability
)

def test_create_payoff_matrix():
    player1_payoffs = [[3, 1], [0, 2]]
    player2_payoffs = [[3, 0], [1, 2]]
    p1_matrix, p2_matrix = create_payoff_matrix(player1_payoffs, player2_payoffs)

    assert np.array_equal(p1_matrix, np.array(player1_payoffs))
    assert np.array_equal(p2_matrix, np.array(player2_payoffs))

    # Test for invalid input
    with pytest.raises(ValueError):
        create_payoff_matrix([[1, 2], [3]], [[1, 2], [3, 4]])

    # Test for 3x3 payoff matrix
    player1_payoffs_3x3 = [[3, 1, 2], [0, 2, 1], [1, 0, 3]]
    player2_payoffs_3x3 = [[3, 0, 1], [1, 2, 0], [2, 1, 3]]
    p1_matrix_3x3, p2_matrix_3x3 = create_payoff_matrix(player1_payoffs_3x3, player2_payoffs_3x3)
    assert np.array_equal(p1_matrix_3x3, np.array(player1_payoffs_3x3))
    assert np.array_equal(p2_matrix_3x3, np.array(player2_payoffs_3x3))

    # Test for error handling when input matrices have different shapes
    with pytest.raises(ValueError):
        create_payoff_matrix([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]])

def test_find_pure_strategy_nash_equilibria():
    player1_matrix = np.array([[3, 1], [0, 2]])
    player2_matrix = np.array([[3, 0], [1, 2]])

    equilibria = find_pure_strategy_nash_equilibria(player1_matrix, player2_matrix)
    assert equilibria == [(0, 0)]

    # Test for no pure strategy Nash equilibrium
    player1_matrix = np.array([[0, 1], [1, 0]])
    player2_matrix = np.array([[1, 0], [0, 1]])
    equilibria = find_pure_strategy_nash_equilibria(player1_matrix, player2_matrix)
    assert equilibria == []

    # Test for 3x3 game
    player1_matrix_3x3 = np.array([[3, 1, 2], [0, 2, 1], [1, 0, 3]])
    player2_matrix_3x3 = np.array([[3, 0, 1], [1, 2, 0], [2, 1, 3]])
    equilibria_3x3 = find_pure_strategy_nash_equilibria(player1_matrix_3x3, player2_matrix_3x3)
    assert (0, 0) in equilibria_3x3
    assert (2, 2) in equilibria_3x3

def test_find_mixed_strategy_nash_equilibrium():
    player1_matrix = np.array([[1, -1], [-1, 1]])
    player2_matrix = np.array([[-1, 1], [1, -1]])

    p1_strategy, p2_strategy = find_mixed_strategy_nash_equilibrium(player1_matrix, player2_matrix)

    assert np.allclose(p1_strategy, [0.5, 0.5], atol=1e-6)
    assert np.allclose(p2_strategy, [0.5, 0.5], atol=1e-6)

    # Test for pure strategy equilibrium
    player1_matrix = np.array([[3, 1], [0, 2]])
    player2_matrix = np.array([[3, 0], [1, 2]])
    p1_strategy, p2_strategy = find_mixed_strategy_nash_equilibrium(player1_matrix, player2_matrix)
    assert np.allclose(p1_strategy, [1, 0], atol=1e-6)
    assert np.allclose(p2_strategy, [1, 0], atol=1e-6)

    # Test for 3x3 game
    player1_matrix_3x3 = np.array([[3, 1, 2], [0, 2, 1], [1, 0, 3]])
    player2_matrix_3x3 = np.array([[3, 0, 1], [1, 2, 0], [2, 1, 3]])
    p1_strategy_3x3, p2_strategy_3x3 = find_mixed_strategy_nash_equilibrium(player1_matrix_3x3, player2_matrix_3x3)
    assert len(p1_strategy_3x3) == 3
    assert len(p2_strategy_3x3) == 3
    assert np.isclose(np.sum(p1_strategy_3x3), 1)
    assert np.isclose(np.sum(p2_strategy_3x3), 1)

def test_analyze_nash_equilibrium_stability():
    player1_matrix = np.array([[3, 1], [0, 2]])
    player2_matrix = np.array([[3, 0], [1, 2]])
    equilibrium = (np.array([1, 0]), np.array([1, 0]))

    analysis = analyze_nash_equilibrium_stability(player1_matrix, player2_matrix, equilibrium)

    assert analysis['stability'] == "Stable"
    assert np.isclose(analysis['player1_payoff'], 3)
    assert np.isclose(analysis['player2_payoff'], 3)
    assert analysis['player1_better_response'] == False
    assert analysis['player2_better_response'] == False

    # Test for unstable equilibrium
    player1_matrix = np.array([[0, 1], [1, 0]])
    player2_matrix = np.array([[1, 0], [0, 1]])
    equilibrium = (np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    analysis = analyze_nash_equilibrium_stability(player1_matrix, player2_matrix, equilibrium)
    assert analysis['stability'] == "Unstable"

    # Test for 3x3 game
    player1_matrix_3x3 = np.array([[3, 1, 2], [0, 2, 1], [1, 0, 3]])
    player2_matrix_3x3 = np.array([[3, 0, 1], [1, 2, 0], [2, 1, 3]])
    equilibrium_3x3 = (np.array([1, 0, 0]), np.array([1, 0, 0]))
    analysis_3x3 = analyze_nash_equilibrium_stability(player1_matrix_3x3, player2_matrix_3x3, equilibrium_3x3)
    assert analysis_3x3['stability'] == "Stable"
    assert np.isclose(analysis_3x3['player1_payoff'], 3)
    assert np.isclose(analysis_3x3['player2_payoff'], 3)

if __name__ == "__main__":
    pytest.main()

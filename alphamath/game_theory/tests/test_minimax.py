import pytest
import numpy as np
from alphamath.game_theory.minimax import create_game_tree, minimax, minimax_alpha_beta, get_optimal_move

def test_create_game_tree():
    payoff_matrix = np.array([[3, 2], [1, 4]])
    game_tree = create_game_tree(payoff_matrix, 2)

    assert game_tree["type"] == "internal"
    assert game_tree["player"] == 0
    assert len(game_tree["children"]) == 2

    for child in game_tree["children"]:
        assert "action" in child
        assert "node" in child
        assert child["node"]["type"] == "internal"
        assert child["node"]["player"] == 1
        assert len(child["node"]["children"]) == 2

        for grandchild in child["node"]["children"]:
            assert grandchild["node"]["type"] == "leaf"
            assert np.array_equal(grandchild["node"]["value"], payoff_matrix)

def test_create_game_tree_invalid_input():
    with pytest.raises(ValueError):
        create_game_tree(np.array([[1]]), 0)  # Invalid depth
    with pytest.raises(ValueError):
        create_game_tree(np.array([1, 2, 3]), 2)  # Invalid payoff matrix shape

def test_minimax():
    payoff_matrix = np.array([[3, 2], [1, 4]])
    game_tree = create_game_tree(payoff_matrix, 2)

    assert minimax(game_tree, 2, True) == 3
    assert minimax(game_tree, 2, False) == 2

def test_minimax_alpha_beta():
    payoff_matrix = np.array([[3, 2], [1, 4]])
    game_tree = create_game_tree(payoff_matrix, 2)

    assert minimax_alpha_beta(game_tree, 2, float('-inf'), float('inf'), True) == 3
    assert minimax_alpha_beta(game_tree, 2, float('-inf'), float('inf'), False) == 2

def test_get_optimal_move():
    payoff_matrix = np.array([[3, 2], [1, 4]])
    game_tree = create_game_tree(payoff_matrix, 2)

    assert get_optimal_move(game_tree) == 0
    assert get_optimal_move(game_tree, use_alpha_beta=False) == 0

    # Test with a different payoff matrix
    payoff_matrix = np.array([[1, 2], [3, 4]])
    game_tree = create_game_tree(payoff_matrix, 2)

    assert get_optimal_move(game_tree) == 1
    assert get_optimal_move(game_tree, use_alpha_beta=False) == 1

def test_minimax_edge_cases():
    # Test with a single-node tree
    leaf_node = {"type": "leaf", "value": np.array([[5]])}
    assert minimax(leaf_node, 0, True) == 5
    assert minimax(leaf_node, 0, False) == 5

    # Test with a tree of depth 1
    simple_tree = {
        "type": "internal",
        "player": 0,
        "children": [
            {"action": 0, "node": {"type": "leaf", "value": np.array([[3]])}},
            {"action": 1, "node": {"type": "leaf", "value": np.array([[1]])}}
        ]
    }
    assert minimax(simple_tree, 1, True) == 3
    assert minimax(simple_tree, 1, False) == 1

def test_minimax_alpha_beta_pruning():
    # Create a game tree where alpha-beta pruning should occur
    complex_tree = {
        "type": "internal",
        "player": 0,
        "children": [
            {"action": 0, "node": {
                "type": "internal",
                "player": 1,
                "children": [
                    {"action": 0, "node": {"type": "leaf", "value": np.array([[3]])}},
                    {"action": 1, "node": {"type": "leaf", "value": np.array([[2]])}}
                ]
            }},
            {"action": 1, "node": {
                "type": "internal",
                "player": 1,
                "children": [
                    {"action": 0, "node": {"type": "leaf", "value": np.array([[5]])}},
                    {"action": 1, "node": {"type": "leaf", "value": np.array([[1]])}}
                ]
            }}
        ]
    }

    # The result should be the same, but alpha-beta should prune some branches
    assert minimax(complex_tree, 2, True) == minimax_alpha_beta(complex_tree, 2, float('-inf'), float('inf'), True)

def test_larger_game_tree():
    payoff_matrix = np.array([[3, 2, 1], [1, 4, 5], [2, 3, 6]])
    game_tree = create_game_tree(payoff_matrix, 3)

    assert game_tree["type"] == "internal"
    assert game_tree["player"] == 0
    assert len(game_tree["children"]) == 3

    minimax_result = minimax(game_tree, 3, True)
    alpha_beta_result = minimax_alpha_beta(game_tree, 3, float('-inf'), float('inf'), True)
    assert minimax_result == alpha_beta_result

    optimal_move = get_optimal_move(game_tree)
    assert optimal_move in [0, 1, 2]

if __name__ == "__main__":
    pytest.main()

import numpy as np
from scipy.optimize import linprog

def create_payoff_matrix(player1_payoffs, player2_payoffs):
    """
    Create a payoff matrix for a two-player game.

    :param player1_payoffs: List of lists representing Player 1's payoffs
    :param player2_payoffs: List of lists representing Player 2's payoffs
    :return: Tuple of numpy arrays (player1_matrix, player2_matrix)
    """
    return np.array(player1_payoffs), np.array(player2_payoffs)

def find_pure_strategy_nash_equilibria(player1_matrix, player2_matrix):
    """
    Find all pure strategy Nash equilibria in a two-player game.

    :param player1_matrix: Numpy array representing Player 1's payoff matrix
    :param player2_matrix: Numpy array representing Player 2's payoff matrix
    :return: List of tuples representing Nash equilibria as (row, col)
    """
    nash_equilibria = []
    rows, cols = player1_matrix.shape

    for i in range(rows):
        for j in range(cols):
            if (player1_matrix[i, j] == np.max(player1_matrix[:, j]) and
                player2_matrix[i, j] == np.max(player2_matrix[i, :])):
                nash_equilibria.append((i, j))

    return nash_equilibria

def find_mixed_strategy_nash_equilibrium(player1_matrix, player2_matrix):
    """
    Find a mixed strategy Nash equilibrium in a two-player game using linear programming.

    :param player1_matrix: Numpy array representing Player 1's payoff matrix
    :param player2_matrix: Numpy array representing Player 2's payoff matrix
    :return: Tuple of numpy arrays (player1_strategy, player2_strategy)
    """
    rows, cols = player1_matrix.shape

    # Solve for Player 2's strategy
    c = np.ones(cols)
    A_ub = -player1_matrix.T
    b_ub = -np.ones(rows)
    A_eq = np.ones((1, cols))
    b_eq = np.ones(1)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    player2_strategy = res.x / np.sum(res.x)

    # Solve for Player 1's strategy
    c = np.ones(rows)
    A_ub = -player2_matrix
    b_ub = -np.ones(cols)
    A_eq = np.ones((1, rows))
    b_eq = np.ones(1)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    player1_strategy = res.x / np.sum(res.x)

    return player1_strategy, player2_strategy

def analyze_nash_equilibrium_stability(player1_matrix, player2_matrix, equilibrium):
    """
    Analyze the stability of a Nash equilibrium.

    :param player1_matrix: Numpy array representing Player 1's payoff matrix
    :param player2_matrix: Numpy array representing Player 2's payoff matrix
    :param equilibrium: Tuple representing the equilibrium strategy (player1_strategy, player2_strategy)
    :return: Dictionary containing stability analysis results
    """
    player1_strategy, player2_strategy = equilibrium
    rows, cols = player1_matrix.shape

    # Calculate expected payoffs
    player1_payoff = np.dot(np.dot(player1_strategy, player1_matrix), player2_strategy)
    player2_payoff = np.dot(np.dot(player1_strategy, player2_matrix), player2_strategy)

    # Check for better responses
    player1_better_response = False
    player2_better_response = False

    for i in range(rows):
        if np.dot(player1_matrix[i], player2_strategy) > player1_payoff:
            player1_better_response = True
            break

    for j in range(cols):
        if np.dot(player1_strategy, player2_matrix[:, j]) > player2_payoff:
            player2_better_response = True
            break

    stability = "Stable" if not (player1_better_response or player2_better_response) else "Unstable"

    return {
        "stability": stability,
        "player1_payoff": player1_payoff,
        "player2_payoff": player2_payoff,
        "player1_better_response": player1_better_response,
        "player2_better_response": player2_better_response
    }

# Example usage
if __name__ == "__main__":
    # Prisoner's Dilemma payoff matrices
    player1_payoffs = [[3, 0], [5, 1]]
    player2_payoffs = [[3, 5], [0, 1]]

    player1_matrix, player2_matrix = create_payoff_matrix(player1_payoffs, player2_payoffs)

    print("Pure Strategy Nash Equilibria:")
    pure_equilibria = find_pure_strategy_nash_equilibria(player1_matrix, player2_matrix)
    for eq in pure_equilibria:
        print(f"({eq[0]}, {eq[1]})")

    print("\nMixed Strategy Nash Equilibrium:")
    mixed_equilibrium = find_mixed_strategy_nash_equilibrium(player1_matrix, player2_matrix)
    print(f"Player 1 strategy: {mixed_equilibrium[0]}")
    print(f"Player 2 strategy: {mixed_equilibrium[1]}")

    print("\nEquilibrium Stability Analysis:")
    stability_analysis = analyze_nash_equilibrium_stability(player1_matrix, player2_matrix, mixed_equilibrium)
    print(f"Stability: {stability_analysis['stability']}")
    print(f"Player 1 Payoff: {stability_analysis['player1_payoff']}")
    print(f"Player 2 Payoff: {stability_analysis['player2_payoff']}")

import numpy as np

def create_game_tree(payoff_matrix, depth):
    """
    Create a game tree from a payoff matrix.

    :param payoff_matrix: 2D numpy array representing the payoff matrix
    :param depth: Maximum depth of the game tree
    :return: Dictionary representing the game tree
    """
    def build_tree(current_depth, player):
        if current_depth == depth:
            return {"type": "leaf", "value": payoff_matrix}

        node = {"type": "internal", "player": player, "children": []}
        for action in range(payoff_matrix.shape[player]):
            child = build_tree(current_depth + 1, 1 - player)
            node["children"].append({"action": action, "node": child})

        return node

    return build_tree(0, 0)

def minimax(node, depth, maximizing_player):
    """
    Implement the minimax algorithm.

    :param node: Current node in the game tree
    :param depth: Current depth in the game tree
    :param maximizing_player: Boolean indicating if it's the maximizing player's turn
    :return: The optimal value for the current player
    """
    if node["type"] == "leaf" or depth == 0:
        return node["value"][0, 0] if maximizing_player else node["value"][1, 0]

    if maximizing_player:
        value = float('-inf')
        for child in node["children"]:
            value = max(value, minimax(child["node"], depth - 1, False))
        return value
    else:
        value = float('inf')
        for child in node["children"]:
            value = min(value, minimax(child["node"], depth - 1, True))
        return value

def minimax_alpha_beta(node, depth, alpha, beta, maximizing_player):
    """
    Implement the minimax algorithm with alpha-beta pruning.

    :param node: Current node in the game tree
    :param depth: Current depth in the game tree
    :param alpha: Alpha value for pruning
    :param beta: Beta value for pruning
    :param maximizing_player: Boolean indicating if it's the maximizing player's turn
    :return: The optimal value for the current player
    """
    if node["type"] == "leaf" or depth == 0:
        return node["value"][0, 0] if maximizing_player else node["value"][1, 0]

    if maximizing_player:
        value = float('-inf')
        for child in node["children"]:
            value = max(value, minimax_alpha_beta(child["node"], depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for child in node["children"]:
            value = min(value, minimax_alpha_beta(child["node"], depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def get_optimal_move(game_tree, use_alpha_beta=True):
    """
    Determine the optimal move for the current player.

    :param game_tree: The game tree
    :param use_alpha_beta: Boolean indicating whether to use alpha-beta pruning
    :return: The optimal action to take
    """
    best_value = float('-inf')
    best_action = None

    for child in game_tree["children"]:
        if use_alpha_beta:
            value = minimax_alpha_beta(child["node"], len(game_tree["children"]) - 1, float('-inf'), float('inf'), False)
        else:
            value = minimax(child["node"], len(game_tree["children"]) - 1, False)

        if value > best_value:
            best_value = value
            best_action = child["action"]

    return best_action

# Example usage
if __name__ == "__main__":
    payoff_matrix = np.array([[3, 2], [1, 4]])
    game_tree = create_game_tree(payoff_matrix, 2)
    optimal_move = get_optimal_move(game_tree)
    print(f"Optimal move: {optimal_move}")

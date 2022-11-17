from copy import deepcopy


def minimax_alpha_beta(game, depth, alpha, beta, max_player):
    if depth == 0 or len(game.get_moves()) == 0:
        return game.state.is_finished(), game

    if max_player:
        best_move = None
        max_value = float('-inf')

        for move in game.get_moves():
            current_game = deepcopy(game)
            current_game.make_move(move)
            value, _ = minimax_alpha_beta(current_game, depth-1, alpha, beta, False)

            if value > max_value:
                max_value = value
                best_move = move

            if max_value >= beta:
                break

            alpha = max(alpha, max_value)

        return max_value, best_move

    else:
        best_move = None
        min_value = float('inf')

        for move in game.get_moves():
            current_game = deepcopy(game)
            current_game.make_move(move)

            value, _ = minimax_alpha_beta(current_game, depth - 1, alpha, beta, True)

            if value < min_value:
                min_value = value
                best_move = move

            if min_value <= alpha:
                break

            beta = min(beta, min_value)

        return min_value, best_move


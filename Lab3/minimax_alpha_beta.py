from copy import deepcopy
import random
from get_heuristic_score import get_heuristic_score


def minimax_alpha_beta(game, depth, alpha, beta, max_player):
    if depth == 0 or game.state.is_finished():
        return [get_heuristic_score(game, max_player), 0]

    if max_player:
        max_value = float('-inf')
        best_moves = {}

        for move in game.get_moves():
            current_game = deepcopy(game)
            current_game.make_move(move)
            [value, _ ] = minimax_alpha_beta(current_game, depth - 1, alpha, beta, False)

            if value > max_value:
                max_value = value
                best_moves = {f"{max_value}": [move]}

            if value == max_value:
                best_moves[f"{max_value}"].append(move)

            if max_value >= beta:
                break

            alpha = max(alpha, max_value)

        return [max_value, random.choice(best_moves[f"{max_value}"])]

    else:
        min_value = float('inf')
        best_moves = {}

        for move in game.get_moves():
            current_game = deepcopy(game)
            current_game.make_move(move)

            [value, _ ] = minimax_alpha_beta(current_game, depth - 1, alpha, beta, True)

            if value < min_value:
                min_value = value
                best_moves = {f"{min_value}": [move]}

            if value == min_value:
                best_moves[f"{min_value}"].append(move)

            if min_value <= alpha:
                break

            beta = min(beta, min_value)

        return [min_value, random.choice(best_moves[f"{min_value}"])]

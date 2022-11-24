from Lab3.minimax_alpha_beta import minimax_alpha_beta
from two_player_games.games.Pick import Pick


def game_simulation(n=4, first_player_depth=1, second_player_depth=1):
    game = Pick(n=n)
    turn = 1

    while game.is_finished() is False:
        print(f"Turn {turn} ------------------------------------------------------------------------------------------")

        [heuristic_score, move] = minimax_alpha_beta(game, first_player_depth, float('-inf'), float('inf'), True)
        game.make_move(move)

        print(f"Player {game.state._other_player.char}: selected move = {move.number}, heuristic_score = {heuristic_score}, player moves = {game.state.other_player_numbers}")

        if game.is_finished():
            break

        [heuristic_score, move] = minimax_alpha_beta(game, second_player_depth, float('-inf'), float('inf'), True)
        game.make_move(move)

        print(f"Player {game.state._other_player.char}: selected move = {move.number}, heuristic_score = {heuristic_score}, player moves = {game.state.other_player_numbers}")

        turn += 1

    if game.state.get_winner():
        print(f"Winner = Player {game.state.get_winner().char}")
    else:
        print(f"Draw")


if __name__ == "__main__":
    game_simulation(n=4, first_player_depth=2, second_player_depth=2)


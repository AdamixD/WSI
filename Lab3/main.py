from two_player_games.games.Pick import Pick
from two_player_games.player import Player
from two_player_games.minimax_alpha_beta import minimax_alpha_beta


def my_game(simulation=False):
    game = Pick(Player("A"), Player("B"))
    print(game)

    while game.is_finished() is False:
        if game.state.get_current_player() == game.first_player:
            value, move = minimax_alpha_beta(game, 3, float('-inf'), float('inf'), True)
            game.make_move(move)
        else:
            if simulation:
                value, move = minimax_alpha_beta(game, 3, float('-inf'), float('inf'), False)
                game.make_move(move)
            else:
                moves = game.get_moves()
                print("\nYour options\n")

                for i, move in enumerate(moves, start=0):
                    print(f"{i}: {move}")

                option = int(input("\nEnter an option: "))
                move = moves[option]
                game.make_move(move)

        print("\n")
        print(game)

    print(f'{(game.get_winner())}')


if __name__ == "__main__":
    my_game()

def get_heuristic_score(game, maximizing_player):
    if not game.is_finished():
        heuristic_score = 0
        potential_combinations = []

        player_moves = game.state.current_player_numbers
        potential_moves = [item.number for item in game.state.get_moves()]

        for combination in game.game_combinations:
            invalid_combination = False
            for item in combination:
                invalid_combination = True if item not in potential_moves and item not in player_moves else False
                if invalid_combination:
                    break
            if not invalid_combination:
                potential_combinations.append(combination)

        for combination in potential_combinations:
            combination_score = 0
            for number in combination:
                combination_score += 1 if number in player_moves else 0

            heuristic_score += combination_score

        return -heuristic_score if maximizing_player else heuristic_score
    else:
        if game.get_winner():
            return -100000 if maximizing_player else 100000
        else:
            return 0

import chess

from neural_network import NeuralNetwork
from config import AlphaZeroConfig
from game_generator import find_move
from game import Game

def play_one_challenge(gen1, gen2, config):
    '''""""""pit for 2 models""""""

    simulating n games, counting how many wins for each player (games which did not end will be ignored)

    returns winner NeuralNetwork class and boolean if given win rate was reached
    """"""'''

    model_old = NeuralNetwork(config)
    model_old.load_weights(gen1)
    model_new = NeuralNetwork(config)
    model_new.load_weights(gen2)
    old_wins = 0
    new_wins = 0

    terminal_values = []
    games_played = []



    for i in range(config.pit_game_num):
        if i % 2 == 1:
            player_white = model_old
            player_black = model_new
        else:
            player_white = model_new
            player_black = model_old

        game = Game()
        # simulate a game
        while not game.is_game_over() and len(game.history) < config.max_moves:
            if game.turn == chess.WHITE:
                action, _ = find_move(game, player_white)
                game.make_move(action)
            else:
                action, _ = find_move(game, player_black)
                game.make_move(action)

        terminal_value = game.terminal_value(len(game.history) - 1)

        # for debug purposes
        terminal_values.append(terminal_value)
        games_played.append(game)

    return games_played
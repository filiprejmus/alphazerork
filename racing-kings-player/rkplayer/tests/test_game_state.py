import unittest

import chess
import numpy as np
from numpy.testing import assert_array_equal
from rkplayer.game import GameState


class TestGameState(unittest.TestCase):

    def test_racing_kings_board_starting_state(self):
        racing_kings_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [-5, -3, -2, -1, 1, 2, 3, 5],
                                       [-4, -3, -2, -1, 1, 2, 3, 4]])

        game_state = GameState(racing_kings_board)

        black = []
        # black figures
        expected = np.zeros((8, 8))
        expected[6, 3] = 1
        expected[7, 3] = 1
        black.append(expected)

        expected = np.zeros((8, 8))
        expected[6, 2] = 1
        expected[7, 2] = 1
        black.append(expected)

        expected = np.zeros((8, 8))
        expected[6, 1] = 1
        expected[7, 1] = 1
        black.append(expected)

        expected = np.zeros((8, 8))
        expected[7, 0] = 1
        black.append(expected)

        expected = np.zeros((8, 8))
        expected[6, 0] = 1
        black.append(expected)

        white = []
        # white figures
        expected = np.zeros((8, 8))
        expected[6, 4] = 1
        expected[7, 4] = 1
        white.append(expected)

        expected = np.zeros((8, 8))
        expected[6, 5] = 1
        expected[7, 5] = 1
        white.append(expected)

        expected = np.zeros((8, 8))
        expected[6, 6] = 1
        expected[7, 6] = 1
        white.append(expected)

        expected = np.zeros((8, 8))
        expected[7, 7] = 1
        white.append(expected)

        expected = np.zeros((8, 8))
        expected[6, 7] = 1
        white.append(expected)
        assert_array_equal(black + white, game_state.get_piece_type_positions_by_color(chess.BLACK))
        assert_array_equal(white + black, game_state.get_piece_type_positions_by_color(chess.WHITE))


if __name__ == '__main__':
    unittest.main()

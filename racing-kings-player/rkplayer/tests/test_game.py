import unittest
import collections
import numpy as np

from rkplayer.game import Game


class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game()

    def test_terminal(self):
        self.assertFalse(self.game.is_game_over())

    def test_3fold_state_repetition(self):
        """Test 3 fold state repetition."""
        for _ in range(2):
            self.game.make_move("h2h3")
            self.game.make_move("a2a3")
            self.game.make_move("h3h2")
            self.game.make_move("a3a2")
        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.terminal_value(), 0)

    def test_repetition_bits_dummy(self):
        """Test repetition bits in image."""
        for _ in range(2):
            self.game.make_move("h2h3")
            self.game.make_move("a2a3")
            self.game.make_move("h3h2")
            self.game.make_move("a3a2")
        # start: rep = 1
        self.assertEqual(self.game.debug_image(0)[0, -1], 64.0)
        # rep = 2
        self.assertEqual(self.game.debug_image(-5)[0, -2], 64.0)
        # rep = 3
        self.assertEqual(self.game.debug_image(-1)[0, -1], 64.0)
        self.assertEqual(self.game.debug_image(-1)[0, -2], 64.0)

        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.terminal_value(-1), 0)

    def test_second_repetition_bit(self):
        """Test repetition counter randomly."""
        most_common_val = 0
        # play games until a game with 2 fold repetition
        while most_common_val < 2:
            self.game.random_move(500)

            cnt = collections.Counter(self.game.history_board)

            most_common = cnt.most_common(1)[0]
            most_common_val = most_common[1]

        # check if repetition equals 2
        idx2 = np.where(self.game.history_board == most_common[0])[0][1]
        image = self.game.make_image(idx2, channel_type="n")
        self.assertEqual(sum(image[13].ravel()), 64.0)
        self.assertEqual(sum(image[14].ravel()), 0)

    def test_fen_encoding(self):
        """Test fen"""
        self.game.random_move(50)
        fen = self.game.fen()
        fen_split = fen.split(" ")
        image = self.game.make_image(-1, channel_type="n")
        self.assertEqual(int(fen_split[4]), int(np.mean(image[1])))
        self.assertEqual(int(fen_split[5]), int(np.mean(image[2])))
        self.assertEqual(1 if fen_split[1] == "w" else 0,
                         int(np.mean(image[0])))


if __name__ == '__main__':
    unittest.main()

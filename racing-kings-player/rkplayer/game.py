import chess.variant
from dataclasses import dataclass
import numpy as np
import os
from copy import deepcopy

from utils import empty_copy

FIGDICT = {'N': 1, 'n': -1, 'B': 2, 'b': -2, 'R': 3,
           'r': -3, 'Q': 4, 'q': -4, 'K': 5, 'k': -5}

STARTFEN = '8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1'


class GameState(object):
    """
        Class to create NN input stack of numpy arrays from the current board state given as array.

        Attributes
        ----------
        __white : numpy array
            stack containing numpy arrays with each array representing a white piece.
        __black : numpy array
            stack containing numpy arrays with each array representing a black piece.
    """

    def __init__(self, board: np.ndarray):
        """

        Parameters
        ----------
        board:
            board that represents the position of the pieces
        """
        board_by_piece_type = {}
        # separate board_array into single pieces with each piece saved as (8,8) array
        # having 1 in fields where the piece is present otherwise 0
        for piece_type_of_player in FIGDICT.values():  # separate board_array into single pieces
            board_by_piece_type[piece_type_of_player] = np.where(board == piece_type_of_player, 1, 0)

        # get white and black pieces as stack of (8,8) arrays
        self.__white = np.array([v for k, v in board_by_piece_type.items() if k > 0])
        self.__black = np.array([v for k, v in board_by_piece_type.items() if k < 0])

    def get_piece_type_positions_by_color(self, leading_color: chess.Color):
        """Returns (10, 8, 8) array containing the positions of every piece type by every color.

        5 8 by 8 arrays each color

        Args:
          leading_color: Color whose positions should appear first in the array

        Returns:
          Positions of every piece type by every color
        """

        # layers with pieces of current player will be stacked on layers of other player
        if leading_color == chess.WHITE:
            return np.vstack((self.__white, self.__black))
        elif leading_color == chess.BLACK:
            return np.vstack((self.__black, self.__white))


def make_repetition_bits(board_hist, board_fen, index):
    """Make 2-bit representation of repetition count."""
    repetitions = np.count_nonzero(board_hist[:index + 1] == board_fen)
    reps_bits = list('{0:03b}'.format(repetitions)[1:3])
    bits = np.array([np.full((8, 8), int(bit), dtype=np.float32)
                     for bit in list(reps_bits)])
    return bits


@dataclass
class Game(chess.variant.RacingKingsBoard):
    """
        A class used to implement game rules and states for SELF PLAY.

        Attributes
        ----------
        history : list
            List to save game history as FEN Strings.
        child_visits : list
            List to store search statistics of game states for MCTS trees.
        game_length : int
            maximal depth of game history before ending.
        possible_racing_kings_moves : list
            List including all game moves.
        num_actions : int
            maximum number of actions in the game.
        move2id : dict
            dictionary mapping moves to ids.
        id2move : dict
            dictionary mapping ids to moves.

        Class Attributes (shared among all class instances)
        ----------
        states : dict
            dictionary that maps FEN states to input stacks for NN, stacks are represented using numpy arrays.
    """

    # initialization of class attributes.
    # note : we used a shared dict among all game instances to save time cause the mapping from game
    # state as string to game state as numpy arrays stays the same for all games.
    states = {}

    def __init__(self, fen=STARTFEN, hist=None, child_visits=None):
        """
            Method for initialisation.

            Parameters
            ----------
            fen : str
                the board's start state in FEN format.
        """
        if hist is not None:
            fen = hist[-1]

        # use super class initialisation from chess.variant.RacingKingsBoard.
        super(Game, self).__init__(fen)

        if hist is not None:
            self.history = hist[:-1]
            self.set_fen(hist[-1])
        else:
            self.history = np.empty(0)

        # initializing some variables
        # self.child_visits = []
        if child_visits is not None:
            self.child_visits = child_visits
        else:
            self.child_visits = np.empty((0, 4096))
        self.game_length = 0

        # load precomputed all possible racing king moves from extern .npy file and save them in a list.
        rkm = str(os.path.join(str(os.path.dirname(__file__)), "possible_racing_kings_moves.npy"))
        self.possible_racing_kings_moves = list(np.load(rkm))

        # calculate maximum number of possible actions/moves.
        self.num_actions = len(self.possible_racing_kings_moves)  # 4096

        # map moves to indices
        self.move2id = dict(zip(self.possible_racing_kings_moves, list(range(self.num_actions))))

        # map indices to moves
        self.id2move = dict(zip(list(range(self.num_actions)), self.possible_racing_kings_moves))

        # store first state and first fen
        self.__add_current_state_to_state_dict()
        self.states["0"] = np.zeros((12, 8, 8))
        self.__add_current_fen_to_history()

    # =========================================================================
    # ==========================       SETUPS         =========================
    # =========================================================================
    def clone(self):
        """
            Method that clones the game. Used for scratch_game in MCTS.

            returned value
            --------
            object: copy of the current class' instance.

            notes
            --------
            As self.states (dict) is shared between all Games 'old' game_states
            will not be recomputed but loaded.
        """
        # copy an instance of the class
        clone = empty_copy(self)

        for k in self.__dict__.keys():
            if k not in ["move_stack", "_stack"]:
                setattr(clone, k, self.__dict__[k])
            else:
                setattr(clone, k, [])

        clone.occupied_co = deepcopy(self.occupied_co)

        return clone

    def remove_random_reset(self, lower_than=3, num_moves=50):
        self.remove_pieces(lower_than)
        self.random_move(num_moves)
        self.__init__(self.fen())

    def remove_pieces(self, lower_than=3):
        """Remove certain pieces.

        Values
        ------------
        6: King
        5: Queen
        4: Rock
        3: Bishop
        2: Knight

        Example
        ------------
        lower_than = 3 -> all Knights are removed
        lower_than = 4 -> Knights and Bishops are removed
        """
        dict_cpy = self.piece_map().copy()
        for pos, piece in dict_cpy.items():
            if piece.piece_type < lower_than:
                self._remove_piece_at(pos)

    @property
    def history_board(self):
        """
            Method to calculate history of game states.

            returned value
            --------
            list: board states represented as FEN strings.
        """
        return np.array([fen.split(' ')[0] for fen in self.history])

    # =========================================================================
    # ======================      INPUT GENERATION      =======================
    # =========================================================================
    def board_array(self, board_fen=None):
        """
            Method to Translate board_fen to np array (8, 8).

            Parameters
            ----------
            board_fen : str
                just board state represented as FEN. e.g. '8/8/8/8/8/8/krbnNBRK/qrbnNBRQ'.
        """

        if board_fen is None:
            board_fen = self.board_fen()

        # initialize 8*8=64 1D numpy array
        arr = np.zeros(64)

        # iterate over FEN string and save the integer id of pieces.
        i = 0
        for char in board_fen:
            if char.isdigit():
                i += int(char)
                continue
            elif char == "/":
                continue
            else:
                arr[i] = FIGDICT.get(char)
                i += 1

        # reshape 1D (64) array to 2D (8,8) array
        return arr.reshape(8, 8)

    def make_image(self, state_index: int = -1, T=8,
                   channel_type="last", repetition=True):
        """
            Method to prepare final NN input stack from Game using T previous board states.

            Parameters
            ----------
            state_index : int
                index of board state.
            T : int
                number of previous game states to include in the stack.
            channel_type : str
                string that specifies the shape of the returned stack (3+ T*12, 8,8) or (8,8, 3+ T*12).
            repetition : bool
                boolean to include repetition löayers in the stack.

            returned value
            --------
            narray: stack of (8,8) numpy array.
            shape=(3 + T*12, 8, 8) or (8,8, 3+ T*12):
                Header information (3, 8, 8):
                    1. color
                    2. fifty_move
                    3. fullmove_cnt
                Data information (T*12, 8, 8):
                    Game State state_index
                    Game State state_index -1
                    ...
                    Game State state_index T-1
        """
        if state_index < 0:
            state_index = len(self.history) + state_index

        # get data/pieces stacks of the last T game states
        data = self.__make_data_image(state_index, T, repetition)

        # get header stacks
        header = self.__make_game_header(state_index).reshape(3, 8, 8)

        # stack header layers on data layers
        state = np.vstack((header, data))

        if channel_type == "last":
            return np.moveaxis(state, 0, 2)

        return state

    def debug_image(self, state_index: int = -1):
        """Get image in debug mode."""
        image = self.make_image(state_index, channel_type="n")
        return np.array([np.sum(arr) for arr in image])[3:].reshape(8, 12)

    # =========================================================================
    # =====================  TERMINAL VALUE + TRAINING   ======================
    # =========================================================================
    def terminal_value(self, state_index: int = -1):
        """
            Method to get terminal value for current player at state_index.

            Parameters
            ----------
            state_index : int
                index of board state.

            returned value
            --------
            int :
                1 if Win
                0 if Draw
                -1 if Loss
                None if Game not finished
        """

        # TODO: check for other draws?
        if state_index < 0:
            state_index = len(self.history) + state_index

        # get current player at the given state index
        to_play = self.history[state_index].split(" ")[1]

        # get the game score for current player
        if self.result() == '1-0':
            if to_play == "w":
                return 1
            else:
                return -1
        elif self.result() == '0-1':
            if to_play == "w":
                return -1
            else:
                return 1
        elif self.result() == '1/2-1/2':
            return 0
        else:
            return None

    def result(self, claim_draw: bool = False) -> str:

        # check if both kings in the last raw than return draw
        split_fen = self.board_fen().split("/")
        if 'K' in split_fen[0] and 'k' in split_fen[0]:
            return '1/2-1/2'

        # otherwise use result() method of super class
        return super().result()

    def make_target(self, state_index: int):
        """Get data to update NN for state_index.

        Returns (from perspective of player with turn at state_index):
            final value of game (-1, 0, 1)
            child_visits
        """
        tv = self.terminal_value(state_index)
        if tv is None:
            tv = 0
        return tv, self.child_visits[state_index]

    def make_policy_target(self, state_index: int):
        """
            Method to get game result and the visits' statistics for the visited game states during the simulation.

            Parameters
            ----------
            state_index : int
                index of board state.

            returned value
            --------
            this method returns :
                b : list containing the visited game states during the simulation and their statistics (how many visit for each state and so on...).
        """
        return self.child_visits[state_index]

    def make_value_target(self, state_index: int):
        """
            Method to get game value.

            Parameters
            ----------
            state_index : int
                index of board state.

            returned value
            --------
                int: encoding result of the game according to the player plying at the given state index (-1, 0, 1).
        """
        return 0 if self.terminal_value(state_index) is None else self.terminal_value(state_index)

    # =========================================================================
    # ==========================         MCTS         =========================
    # =========================================================================
    def store_search_statistics(self, root):
        """
            Method to update the statistics of the visited game states during MCTS simulation.

            Parameters
            ----------
            root : Node
                Node representing the root game state where the simulation started.
        """

        # get the sum of the simulations starting from the node 'root'
        sum_visits = sum(child.visit_count for child in root.children.values())
        # self.child_visits.append([
        # root.children[a].visit_count / sum_visits if a in root.children else 0
        # for a in range(self.num_actions)
        # ])

        # update the statistics of children states visited from root state
        search_stats = np.array(
            [root.children[a].visit_count / sum_visits if a in root.children else 0 for a in range(self.num_actions)])
        self.child_visits = np.vstack((self.child_visits, search_stats))

    # =========================================================================
    # ==========================        MOVES         =========================
    # =========================================================================
    @property
    def legal_moves_in_uci(self):
        """All legal moves in uci format.

        E.g. 'a1a8', 'e4e5', etc.
        """

        # get all legal moves. 'legal_moves' is inherited attribute from super class that returns all possible moves
        return [m.uci() for m in list(self.legal_moves)]

    def make_move(self, action, validate=True):
        """
            Method to apply given action in current game state and move to next game state.

            Parameters
            ----------
            action : str
                the action to apply.
            validate : bool
                boolean to check validity of action if set to True.
            update_history : bool
                boolean to update game history if set to True.
            update_states : bool
                boolean to update the saved game states dict if set to True.
        """

        # get the action in UCI string format e.g. 'a1a8'
        if isinstance(action, int):
            action = self.id2move.get(action)
        # print("==================ACTION: " + action)

        # check if action is legal
        if validate is True and action not in self.legal_moves_in_uci:
            raise ValueError("Move not legal!, Move: " + action + " | Fen: " + self.fen())

        # save the played action. push() is inherited method from super class to save actions in a stack
        self.push(chess.Move.from_uci(action))

        # update game length
        self.game_length += 1

        # update game history
        self.__add_current_fen_to_history()

    def random_move(self, num_of_moves=1, validate=False):
        """
            Method to apply random moves => FOR TESTING PURPOSES !

            Parameters
            ----------
            num_of_moves : int
                number of moves to make.
            validate : bool
                boolean to check validity of action if set to True.

            returned value
            --------
            None: only if no random moves exists
        """
        for i in range(num_of_moves):
            # check if legal move exists
            if len(self.legal_moves_in_uci) == 0:
                return None
            # get random move
            action = np.random.choice(self.legal_moves_in_uci)
            # apply move
            self.make_move(action, validate=validate)

    # =========================================================================
    # ==========================      DRAW RULES      =========================
    # =========================================================================
    def is_seventyfive_moves(self) -> bool:
        """Checks seventy five rule (Overrides base class to turn seventy five rule into a fifty rule).

        Returns:
            Returns whether 75 (50) rule is violated
        """
        if self.halfmove_clock >= 100:
            if any(self.generate_legal_moves()):
                return True
        return False

    def is_fivefold_repetition(self) -> bool:
        """Checks five fold repetition rule (Overrides base class to turn five fold rule into a three fold rule).

        Returns:
            Returns whether five (three) fold repetition rule is violated
        """
        return self.is_repetition(3)

    def is_repetition(self, num: int = 3) -> bool:
        """Checks if the current position has repeated 3 (or a given number of) times.

        Returns:
            True if the current position has repeated 3 (or a given number of) times.
        """

        if sum(self.history_board == self.board_fen()) == num:
            return True
        return False

    # =========================================================================
    # ==========================        HELPER        =========================
    # =========================================================================
    def move_from_id(self, id):
        """
            Method to get move corresponding to given move id.

            Parameters
            ----------
            id : int
                int representing the move id.

            returned value
            --------
            str: the move in UCI format.
        """
        return self.id2move.get(id)

    def __get_whose_turn_in_history(self, time_index: int) -> chess.Color:
        """Returns the side whose turn it was in the specified time point.

        Args:
            time_index: Point in time (history) that the playing side should be determined from

        Returns:
            Side whose turn it was
        """

        # get player from history
        side = self.history[time_index].split(" ")[1]

        if side == "w":
            return chess.WHITE
        elif side == "b":
            return chess.BLACK

    def __make_data_image(self, state_index: int, T=8, repetition=True):
        """
            Method to Create data image of T previous states starting from current state index.

            Parameters
            ----------
            state_index : int
                index of board state.
            T : int
                number of previous game states to include in the stack.
            repetition : bool
                boolean to include repetition layers in the stack.

            returned value
            --------
            narray: stack of T game states with shape (T*12, 8, 8):
                 ---- Game state at current index (12, 8, 8)---
                 P1:
                 knight (8x8)
                 ...
                 king (8x8)
                 P2
                 knight (8x8)
                 ...
                 king (8x8)
                 ----------------------------------------------
                 Game state at index=state_index - 1 (12, 8, 8)
                 ...
                 Game state at index=state_index - (T-1) (12, 8, 8)
                 ----------------------------------------------
                 Total shape: (T*12, 8, 8)
            If Game_state at index is not available (e.g. state at T-3) an array of np.zeros(12, 8, 8) is appended.
        """
        if state_index < 0:
            state_index = len(self.history) + state_index

        # get the game states to stack from game history
        T_int = min(T, len(self.history), state_index)
        state_strs = self.history[(state_index - T_int + 1):state_index + 1].tolist()
        state_strs.reverse()

        # append first state artefact
        if T_int < T:
            state_strs.append(self.history[0])
        data = []

        # get player on game state with state_index
        turn = self.__get_whose_turn_in_history(state_index)
        board_hist = self.history_board
        # stack the game states.
        for i, s in enumerate(state_strs):
            # get board from fen
            s = s.split(" ")[0]
            try:  # load stored state
                state = self.states.get(s).get(turn)
            except AttributeError:  # create state if not found
                board_array = self.board_array(s)
                gs = GameState(board_array)
                self.states[s] = gs
                state = gs.get_piece_type_positions_by_color(turn)
            data.append(state)
            if repetition is True:
                data.append(make_repetition_bits(board_hist, s, state_index - i))

        data = np.vstack(data)

        # append zeros afterwards
        size = 12 if repetition is True else 10
        while data.shape[0] < (T * size):
            state = self.states.get("0")
            data = np.vstack((data, state))
        return data

    def __add_current_state_to_state_dict(self):
        """Adds the current game state to the state dict."""
        board_fen = self.board_fen()
        if board_fen not in self.states:
            self.states[self.board_fen()] = GameState(self.board_array())

    def __add_current_fen_to_history(self):
        """Adds the current fen to the history."""
        self.history = np.hstack((self.history, self.fen()))

    def __make_game_header(self, state_index: int = -1):
        """
            Method to get game header  of color, halfmove_clock and fullmove_cnt.

            Parameters
            ----------
            state_index : int
                index of board state.

            returned value
            --------
            narray: stack of shape (3, 8, 8) with 3*(8,8) np.full arrays (all ones or all zeros).
        """

        if state_index < 0:
            state_index = len(self.history) + state_index

        # get board_state as FEN string
        fen = self.history[state_index]
        splitfen = fen.split(" ")

        color = np.full((8, 8), int(self.__get_whose_turn_in_history(state_index)), dtype=np.float32)

        # (8,8) array of ones if white is current player otherwise zeros
        fifty_move = np.full((8, 8), int(splitfen[4]), dtype=np.float32)

        # (8,8) array full of number of moves since the first black's move
        fullmove_cnt = np.full((8, 8), int(splitfen[5]), dtype=np.float32)

        # stack the 3 sub headers
        return np.array([color, fifty_move, fullmove_cnt])

import random
from typing import Tuple

import numpy as np
import time
import concurrent
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Pipe
from collections import deque
import os

from neural_network import NeuralNetwork
from game import Game
from config import AlphaZeroConfig, custom_selfplay_plan
from mcts_tree import Node
from shared_storage import SharedStorage
from utils import softmax_sample
import tensorflow as tf
import sys
import signal
import logging

pids = []


class Selfplay:
    """
        A class to realize multiprocessing self plays.

        Attributes
        ----------
        config : object
            object which holds the configurations.
        current_neural_network : object
            object of class NeuralNetwork, most recent model to generate games with.
        manager : Manager from multiprocessing
            manages the list of pipes (communication endpoints to the ModelAPI) for all processes.
        game_buffer : list
            contains all the simulated games.
    """

    def __init__(self):
        """
            Method for initialization.
        """
        self.config = config
        self.current_neural_network = neural_network
        self.manager = Manager()
        self.pipes = self.manager.list(self.current_neural_network.create_pipes(self.config.num_pipes))

        self.game_buffer = np.asarray([])

    def start(self):
        """
            Method that starts the self play workers (processes).

            Note: Each process simulate its own game by exchanging data with the prediction worker.
            Meanwhile parent process waits til one game gets finished simulating after the other in order to
            append the game object to the buffer.
        """
        futures = deque()
        game_id = 0

        # call to summon the processes
        with ProcessPoolExecutor(max_workers=self.config.max_processes) as executor:
            # append job which workers get allocated to, each with their own pipe
            for _ in range(self.config.max_processes):
                futures.append(executor.submit(play_game, cur_pipes=self.pipes))

            while True:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        completed_game = future.result()
                        del futures[futures.index(future)]
                        if completed_game.child_visits.shape[0] == 0:
                            continue
                    except Exception as exc:
                        logging.exception("One process generated the following exception:\n{}".format(exc))
                        del futures[futures.index(future)]
                        continue
                    else:
                        # submit new game run into pipe
                        futures.append(executor.submit(play_game, cur_pipes=self.pipes))

                        game_id += 1
                        logging.info("Game %d has %d moves and ended with value %d"
                                     % (game_id, len(completed_game.history), completed_game.make_value_target(-1)))

                        # prepare game to write into file
                        game = np.empty((2,), dtype=np.ndarray)
                        game[0] = completed_game.history
                        game[1] = completed_game.child_visits
                        game = game[None]

                        try:
                            shared_storage.add_game(game)
                        except Exception:
                            logging.warning("Couldn't add game to game pool")
                            self.stop(executor)

                        game_count = shared_storage.increase_game_count()
                        logging.info("%d games have been simulated" % game_count)

                        # check if enough games have been collected
                        if game_count >= config.window_size:
                            logging.info("Goal number of %d simulated games has been reached."
                                         "Shutdown self play worker" % config.window_size)
                            self.stop(executor)

    def stop(self, executor: ProcessPoolExecutor):
        logging.info("Shutdown selfplay worker")
        shutdown_game_generator_process_pool(executor)
        self.current_neural_network.kill_predictor()
        self.manager.shutdown()
        logging.info("Shutdown selfplay worker - successful")
        exit()


def shutdown_game_generator_process_pool(executor: ProcessPoolExecutor):
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
        logging.info("Kill process with id: " + str(pid))
    executor.shutdown(wait=True)


def play_game(cur_pipes):
    """
        Method to generate a new fresh game and simulate it using MCTS simulations
        and NN inferences till game is finished or max number of moves is reached.
        Will be ran by each self play process.

        Parameters
        ----------
        cur_pipes : List
            list of Pipe object, holds the communication endpoints to the prediction worker

        returned value
        -----------
        game : object
            object of class Game, holds the simulated game.
    """
    logging.info("selfplay with pid %d" % (os.getpid()))
    pids.append(os.getpid())
    # new random seed for each process
    np.random.seed()
    timer = time.time()

    # pop one pipe
    pipe = cur_pipes.pop()

    # generate new game
    game = Game()

    # apply custom setup
    game.remove_random_reset(lower_than=lower_than, num_moves=random.randint(min_rand_moves, num_rand_moves))
    while game.is_game_over():
        game = Game()
        game.remove_random_reset(lower_than=lower_than, num_moves=random.randint(min_rand_moves, num_rand_moves))

    # loop till game is finished or max_moves (max number of moves) reached
    while not game.is_game_over() and len(game.history) < config.max_moves:
        # run mcts simulation for current state and choose action and get node representing the current state
        action, root = find_move(game, pipe)
        # apply chosen action
        game.make_move(action)
        # save statistics for the current simulated game state
        game.store_search_statistics(root)

    # return pipe
    cur_pipes.append(pipe)

    logging.info("one selfplay has been simulated in %d s" % (round(time.time() - timer, 3)))
    return game


def find_move(game: Game, pipe: Pipe) -> Tuple[str, Node]:
    """Uses MCTS simulation to find a fitting chess move inside the game.

    To decide on an action, we run N simulations, always starting at the root
    of the search tree and traversing the tree according to the UCB formula
    until we reach a leaf node.

    Starts with root node of 'game' and expands children with added exploration
    noise.

    Args:
      game:
        Game in which we find the optimal move.
      pipe:
        TODO

    Returns:
      The move and the root node of the MCT
    """
    root = Node(0)
    evaluate(root, game, pipe)
    add_exploration_noise(root)

    for _ in range(config.num_simulations):
        # Set current node to root node
        node = root
        # clone game
        scratch_game = game.clone()
        search_path = [node]

        # While children of node exist:
        while node.expanded():
            # select action and child in greedy way and apply action to cloned game
            action, node = select_child(node)
            scratch_game.make_move(action)
            search_path.append(node)
        # Calculate a value for node of cloned game by evaluating it with 'network'
        value = evaluate(node, scratch_game, pipe)
        backpropagate(search_path, value, int(scratch_game.turn))

    return select_action(game, root), root


def select_action(game: Game, root: Node):
    """"
        Method to select real action after the MCTS simulations.

        Parameters
        ----------
        root : object
            the current node(game state).
        game : object
            object of class Game encoding the current game

        returned value
        -----------
        str: the chosen action.
    """

    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    vc, act = zip(*visit_counts)
    if len(game.history) < config.num_sampling_moves:
        max_vc = softmax_sample(vc)
    else:
        max_vc = np.argmax(vc)
    return act[max_vc]


def select_child(node: Node):
    """"
        Method to select next action and next child (game state)config.path_to_game_pool_file
        according to the UCB formula during the MCTS simulation.

        Parameters
        ----------
        node : object
            the current node(game state).

        returned value
        -----------
        Tuple(a,b):
            the chosen action
    """

    _, action, child = max((ucb_score(node, child), action, child)
                           for action, child in node.children.items())
    return action, child


def ucb_score(parent: Node, child: Node):
    """The score for a node is based on its value, plus an exploration bonus
    based on the prior."""
    pb_c = np.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


def evaluate(node: Node, game: Game, pipe: Pipe):
    """
        Method that uses the given NN to predict the value and policy vector
        of the current game state.

        Note: Solely the game history will be sent to the prediction worker.

        Parameters
        ----------
        node : object
            object of class Node encoding the game state to be evaluated.
        game : object
            object of class Game encoding the current game
        pipe
            object of class Pipe from multiprocessing library, stores the communication endpoint to ModelAPI.

        returned value
        -----------
        float: the predicted game state value
    """

    # send current game history to ModelAPI

    try:
        pipe.send(game.history)
    except Exception as exc:
        logging.exception('send pipe error, SelfPlay Process exiting.. Exception: ' + str(exc))
        sys.exit()

    # receive policy vector
    try:
        nn_inference = pipe.recv()
    except Exception as exc:
        logging.exception('recv pipe error, SelfPlay Process exiting.. Exception: ' + str(exc))
        sys.exit()

    # fetch the predicted value
    try:
        value = nn_inference[0].item()
        policy_logits = nn_inference[1].flatten()
    except:
        logging.info('exception in GameGenerator in evaluate in returned inference values')
        sys.exit()
    # get the current player
    node.to_play = int(game.turn)

    # build policy dict that maps legal actions to its probability
    policy = {a: np.exp(policy_logits[a])
              for a in [game.move2id.get(move) for move in game.legal_moves_in_uci]}

    # calculate probabilities just for legal actions
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)

    return value


def backpropagate(search_path, value: float, to_play):
    """
        Method that back propagates the predicted value along the path nodes.

        Parameters
        ----------
        search_path : list
            list of path nodes.
        value : float
            the predicted value of the game state.
        to_play : int
            he current player of the evaluated game state.
    """

    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


def add_exploration_noise(node: Node):
    """Adds dirichlet noise to the prior of the root.

    This encourages the search to explore new actions..

    Args:
      node:
        Root node
    """

    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


if __name__ == '__main__':

    config = AlphaZeroConfig()
    shared_storage = SharedStorage(config)

    gen = shared_storage.current_generation

    game_log_name = 'GameLog_' + str(gen) + '_' + str(os.getpid() + np.random.randint(99999)) + '.log'
    logging.basicConfig(filename=game_log_name, level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Num GPUs Available: %d" % len(tf.config.experimental.list_physical_devices('GPU')))

    if gen == 0:
        neural_network = NeuralNetwork(config)
    else:
        while True:
            model = shared_storage.load_model()

            if model is not None:
                neural_network = NeuralNetwork(config, model)
                break
            else:
                logging.info("No model for generation " + str(gen) + " exists. Trying again...")
                time.sleep(config.timout_poll_new_model)

    logging.info("============starting selfplay with generation: " + str(gen) + "==================")

    lower_than, num_rand_moves = custom_selfplay_plan(gen)
    min_rand_moves = num_rand_moves * 0.5
    Selfplay().start()
    logging.info("________ EXIT ________")
    sys.exit()

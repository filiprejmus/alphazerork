import logging
import os
import time
from pathlib import Path

import numpy as np
from filelock import FileLock
from tensorflow.python.keras import Model, models

from game import Game
from config import AlphaZeroConfig
from utils import append_to_file, save_as_file, load_file


class SharedStorage(object):
    """Storage that is shared between all game generating instances and the training instance.

    Attributes:
        config: TODO.
    """

    # path of shared folder
    __shared_folder_path = '../shared/'
    # path of game pool
    __game_pool_folder_path = __shared_folder_path + 'pool/'
    # path of file which stores information about the current generation
    __current_generation_file_path = __shared_folder_path + 'generation.txt'
    # model folder
    __model_folder_path = __shared_folder_path + 'models/'
    # TensorBoard training logs folder
    __tensor_board_training_logs_folder_path = __shared_folder_path + 'tensor_board_training_logs/'

    def __init__(self, config: AlphaZeroConfig):
        self.config = config

    def save_model(self, model: Model):
        """Saves the given model of the current generation to the shared storage.

        Args:
            model: Model that should be saved
        """
        model_path = self.tf_model_path
        with FileLock(model_path + '.lock'):
            models.save_model(model, model_path)

    def load_model(self) -> Model:
        """Loads the model of the current generation.

        Returns:
            Model of the given generation
        """
        model_path = self.tf_model_path

        logging.info("loading model " + model_path)
        with FileLock(model_path + '.lock'):
            return models.load_model(model_path)

    @property
    def current_generation(self) -> int:
        """The current generation of the shared storage."""
        with FileLock(self.__current_generation_file_path + '.lock'):
            with open(self.__current_generation_file_path, "r") as f:
                return int(f.read())

    def advance_to_next_generation(self) -> int:
        """Advances to one generation after the current one."""
        gen = self.current_generation

        with FileLock(self.__current_generation_file_path + '.lock'):
            with open(self.__current_generation_file_path, "w+") as f:
                next_gen = gen + 1
                f.write(str(next_gen))
                return next_gen

    @property
    def games(self) -> np.ndarray:
        """All games belonging to the current generation"""
        training_set = self.__game_data

        return np.vectorize(
            lambda game, history, child_visits:
            Game(hist=history, child_visits=child_visits))(training_set[:, 0], training_set[:, 0], training_set[:, 1])

    def add_game(self, game: Game):
        """Adds game to the game pool of the current generation.

        Args:
            game: Game to be added to the game pool.
        """
        gen = self.current_generation
        pool_file_path = self.__current_gen_pool_folder_path + 'pool_' + str(os.getpid()) + '_' + str(gen) + '.npz'

        if not Path(pool_file_path).is_file():
            logging.info("Pool file does not exist. Creating new pool file now...")
            save_as_file(pool_file_path, np.empty((0, 2), dtype='U51'))

        # try 10 times to add the game to the game pool
        trials = 10

        for i in range(trials):
            try:
                append_to_file(pool_file_path, game)
                return
            except Exception:
                logging.warning("Couldn't add game to game pool. Trying again...")
                i += 1
                time.sleep(2)
        raise Exception

    @property
    def game_count(self) -> int:
        """Amount of games of the current generation inside the game pool"""
        counter_path = self.__counter_path
        if not self.__game_counter_exists:
            self.__create_game_counter()

        with FileLock(counter_path + '.lock'):
            with open(counter_path, "r") as f:
                return int(f.read())

    def increase_game_count(self) -> int:
        """Increases the game count of the current generation.

        Returns:
            The new game count after increase
        """
        counter_path = self.__counter_path
        if not self.__game_counter_exists:
            self.__create_game_counter()

        with FileLock(counter_path + '.lock'):
            with open(counter_path, "r") as f:
                counter = int(f.read())
            with open(counter_path, 'w') as f:
                counter += 1
                f.write(str(counter))
                return counter

    @property
    def tensor_board_training_logs_path(self) -> str:
        """Path of TensorBoard training logs folder of current generation"""
        return self.__tensor_board_training_logs_folder_path + self.config.modelname + '_gen-' + str(self.current_generation)

    @property
    def tf_model_path(self) -> str:
        return self.__model_folder_path + 'model-' + self.config.modelname + '_gen-' + str(self.current_generation)

    @property
    def trt_model_path(self) -> str:
        return self.__model_folder_path + 'model-' + self.config.modelname + '_trt_gen-' + str(self.current_generation)

    @property
    def __counter_path(self) -> str:
        """Path of the counter file"""
        return self.__game_pool_folder_path + 'counter_' + str(self.current_generation) + '.txt'

    @property
    def __current_gen_pool_folder_path(self) -> str:
        """Path of the folder where the pool files of the current generation lie"""
        return self.__game_pool_folder_path + 'gen_' + str(self.current_generation) + '/'

    @property
    def __game_data(self) -> np.ndarray:
        """Data of all games belonging to the current generation"""
        training_set = np.empty((0, 2), dtype='U51')
        pool_file_path = self.__current_gen_pool_folder_path
        for filename in os.listdir(pool_file_path):
            if filename.endswith(str(self.current_generation) + ".npz"):
                i = 0
                while i < 10:
                    logging.info("loading file %s" % filename)
                    tmp, done = load_file(pool_file_path + filename)
                    if done:
                        training_set = np.vstack((training_set, tmp))
                        break
                    else:
                        logging.warning("Could not load" + filename + "! Trying again...")
                        i += 1
                        time.sleep(4)
                        if i == 10:
                            logging.warning("Could not load" + filename + "! Skipping it.")
                        continue
        return training_set

    def __create_game_counter(self):
        """Create game counter for the current generation and set count to 0"""
        with open(self.__counter_path, "w+") as f:
            f.write("0")

    @property
    def __game_counter_exists(self) -> bool:
        """Boolean indicating whether the game counter for the current generation exists or not"""
        return Path(self.__counter_path).is_file()

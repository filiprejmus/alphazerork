import os
import shutil
import stat
import time
import sys
import logging

import tensorflow as tf
import numpy as np

from collections import defaultdict, deque
from multiprocessing import connection, Pipe
from threading import Thread

from multiprocessing import Process
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from config import AlphaZeroConfig
from game import Game
from filelock import FileLock

from shared_storage import SharedStorage


class ModelAPI:
    """
        A class which sets up and starts the prediction producer.

        Note:
        listens on all pipes:
        - as soon as requests arrive, check if a state is located in the cache
            if yes, return associated data immediately
            else, predict all/the rest of the requests in one batch
        - then send prediction to the self-play workers
        - eventually write the recently predicted states into the cache

        Attributes
        ----------
            config : object
                object which holds the configurations.
            model : tensorflow.keras.model
                most recent model to generate games with.
            infer_model_trt : object
                holds the by TensorRT transformed model.
            pipes : list
                list of Pipe objects, holds two-way communication endpoints to the self play worker.
            cache : object
                object of class Cache.
            gen : int
                information about training generation.
    """

    def __init__(self, model, config: AlphaZeroConfig):
        """
            Method for initialisation.
        """
        self.config = config
        self.model = model
        self.infer_model_trt = None
        self.pipes = []
        self.end = False
        self.worker = None
        self.storage = SharedStorage(config)

    def start(self):
        """
            Method to setup the TensorRT model and to start prediction worker thread.

            Attributes
            ----------
            worker : object
                Thread object to start.
        """

        if self.config.use_tensorRT:
            # build the TensorRT model
            self.build_trt_model()

        # initialize the prediction worker
        self.worker = Thread(target=self._predict, name="prediction_producer")

        # prediction worker will die as soon as main programm shuts down
        self.worker.daemon = True

        # start the prediction worker
        self.worker.start()

    def create_pipe(self):
        """
            Method to create communication pipes.

            returned value
            -----------
            your_end : object
                Pipe object which self play workers store for exchange of information with the prediction worker.
        """
        # create two way communication pipe
        our_end, your_end = Pipe()

        # store for exchange of information with self play workers
        self.pipes.append(our_end)

        return your_end

    def kill(self):
        self.end = True
        logging.info("kill the prediction worker")

    def _predict(self):
        """
            Method to receive game histories from several processes in order to prepare, predict and eventually
            dispatch policy vectors to the process associated with the data.

            Note: Also checks the cache if the FEN has been infered before. If yes, respective policy vector will get
            returned immediately.
        """
        logging.basicConfig(filename='ModelApiLog.log', level=logging.DEBUG)
        breaking = False
        while not self.end:
            # listen and wait for incoming data
            ready = connection.wait(self.pipes, timeout=self.config.timeout_listen)
            if not ready:
                continue

            recv_data, result_pipes, recv_fen = [], [], []

            # collect data from pipes
            for pipe in ready:
                if breaking:
                    break
                while pipe.poll():
                    try:
                        recv_data.append(pipe.recv())
                        result_pipes.append(pipe)
                    except Exception as exc:
                        logging.exception("pipe recv error, exception: " + str(exc))
                        breaking = True
                        break

            # make fens from game histories
            recv_fen = self.make_fen(recv_data)

            # check the cache to which incoming state has already been predicted
            # if all prediction requests got answered, continue
            if len(recv_data) == 0:
                continue

            # image preparation
            data = self.make_image(recv_data)
            data_images = np.asarray(data, dtype=np.float32)

            # predict either with TensorRT model or tf.keras.model
            if self.config.use_tensorRT:
                value, policy = self._infer_trt(data_images)
            else:
                value, policy = self.model.predict_on_batch(data_images)

            for pipe, val, pol, fen in zip(result_pipes, value, policy, recv_fen):
                # send back value and policy vectors
                try:
                    pipe.send((val, pol))
                except Exception as exc:
                    logging.exception("pipe send error, exception: " + str(exc))
                    break

                # write value and policy vectors of associated game state to the cache

        for pipe_to_close in self.pipes:
            pipe_to_close.close()

    def build_trt_model(self):
        """
            Method to build TensorRT from a tf.keras.model.

            Note: Configurations are depend on underlying GPU. Check if your GPU supports specific TensorRT parameters.
        """

        model_dir_tf = self.storage.tf_model_path
        model_dir_trt = self.storage.trt_model_path

        os.umask(0)
        with FileLock(model_dir_trt + '.lock'):
            if not os.path.exists(model_dir_trt):
                # set conversion parameters
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                conversion_params = conversion_params._replace\
                    (max_workspace_size_bytes=self.config.max_workspace_size_bytes)
                conversion_params = conversion_params._replace(
                    maximum_cached_engines=30)
                conversion_params = conversion_params._replace(precision_mode=self.config.precision_mode)
                conversion_params = conversion_params._replace(minimum_segment_size=self.config.minimum_segment_size)

                # initialize the converter
                converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir_tf,
                                                    conversion_params=conversion_params)

                # convert
                converter.convert()
                # save TensorRT model
                if not os.path.exists(model_dir_trt):
                    logging.info("Saving TensorRT Model..")
                    converter.save(model_dir_trt)

        print(f"Load TensorRT Model..  (from: {model_dir_trt})")
        logging.info("Load TensorRT Model..  (from: %s) " % model_dir_tf)

        with FileLock(model_dir_trt + '.lock'):
            model_trt = tf.saved_model.load(model_dir_trt)

        # set attribute to hold TensorRT model
        self.infer_model_trt = model_trt.signatures["serving_default"]

    def _infer_trt(self, data_images):
        """
            Method for the inference on the TensorRT model.

            Parameters
            -----------
            data_images : np.array
                contains all images to be predicted

            returned value
            -----------
            Tuple (a,b):
                a : value vector
                b : policy vector
        """
        output = self.infer_model_trt(tf.constant(data_images, dtype=float))
        return output["value_out"].numpy(), output["policy_out"].numpy()

    @staticmethod
    def make_fen(recv_data):
        """
            Method for unmarshalling the received data to processable FENs

            Parameters
            -----------
            recv_data : array
                contains game histories

            returned value
            -----------
            recv_fen : array
                contains translated FENs
        """
        # initialize scratch game
        recv_fen = []
        for hist in recv_data:
            # set history of scratch game to the received one
            recv_fen.append(hist[-1])
        return recv_fen

    @staticmethod
    def make_image(recv_data):
        """
            Method for unmarshalling the received data to for the model processable images

            Parameters
            -----------
            recv_data : array
                contains game histories

            returned value
            -----------
            images : array
                contains game images ready for inference
        """
        # initialize scratch game
        scratch_game = Game()
        images = []

        for hist in recv_data:
            # set history of scratch game to the received one
            scratch_game.history = hist

            # make image and append to array
            images.append(scratch_game.make_image())
        return images

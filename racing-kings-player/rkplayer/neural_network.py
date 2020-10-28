from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Dense, Flatten, Add, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from config import AlphaZeroConfig
from ModelAPI import ModelAPI
from utils import ReplayBuffer


class NeuralNetwork:
    """The Neural Network used in the game generating and training process.

    Attributes:
        model: Model of the Neural Network
        __config: TODO
        __api: TODO
    """

    def __init__(self, config: AlphaZeroConfig, model=None):
        self.model = model if model is not None else self.__build_model()
        self.__config = config
        self.__api = ModelAPI(self.model, self.__config)

    def create_pipes(self, count):
        self.__api.start()
        return [self.__api.create_pipe() for _ in range(count)]

    def kill_predictor(self):
        self.__api.kill()

    def train(self, replay_buffer: ReplayBuffer, tensor_board_path: str):
        """Trains the neural network.

        Args:
          replay_buffer:
            Buffer holding the training games.
          tensor_board_path:
            Path to folder in which the TensorBoard callback saves its logs
        """
        for i in range(self.__config.training_steps):
            # take training batch from the data pool
            game_vec, value_vec, policy_vec = replay_buffer.sample_batch

            # fit the model to the extracted batch
            self.model.fit(game_vec, [value_vec, policy_vec], epochs=self.__config.epochs, batch_size=self.__config.batch_size,
                           callbacks=[keras.callbacks.TensorBoard(log_dir=tensor_board_path, update_freq='epoch')])

    def __build_model(self):
        """Method for builds a new model.

        Architecture of the new model
        1. layer:
            - 2d convolutional layer
            - batch normalization
            - activation layer (ReLU)

        2. chosen number of residual blocks, each consisting of:
            - 2d convolutional layer
            - batch normalization
            - 2d convolutional layer
            - batch normalization
            - skip connection to previous layer
            - activation layer (ReLU)

        3.1 policy head
            - 2d convolutional layer
            - batch normalization
            - activation (ReLU)
            - flatten
            - dense (output shape: (4096,), activation: "softmax")

        3.2 value head
            - 2d convolutional layer
            - batch normalization
            - activation (ReLU)
            - flatten
            - dense (output shape: (1,), activation: "tanh")

        Returns:
          New model
        """
        # ---------------------------------------------------------------------#
        # -----------------------     input neurons    ------------------------#

        in_x = x = Input((8, 8, 99))

        # ---------------------------------------------------------------------#
        # -----------------     first convolutional layer     -----------------#

        x = Conv2D(filters=self.__config.cnn_filter_num,
                   kernel_size=self.__config.cnn_first_filter_size,
                   padding="same",
                   data_format="channels_last",
                   use_bias=False,
                   kernel_regularizer=l2(self.__config.l2_reg),
                   name="input_conv-" + str(self.__config.cnn_first_filter_size) + "-" + str(self.__config.cnn_filter_num))(
            x)

        x = BatchNormalization(axis=-1,
                               name="input_batchnorm")(x)

        x = Activation("relu",
                       name="input_relu")(x)

        # ---------------------------------------------------------------------#
        # -----------------------     residual layers    ----------------------#

        # create many residual blocks on top of each other
        for i in range(self.__config.res_layer_num):
            x = self.__add_residual_layer(x, i + 1)
        res_out = x

        # ---------------------------------------------------------------------#
        # -------------------------     policy head    ------------------------#

        x = Conv2D(filters=2,
                   kernel_size=1,
                   data_format="channels_last",
                   use_bias=False,
                   kernel_regularizer=l2(self.__config.l2_reg),
                   name="policy_conv-1-2")(res_out)

        x = BatchNormalization(axis=-1,
                               name="policy_batchnorm")(x)

        x = Activation("relu",
                       name="policy_relu")(x)

        x = Flatten(name="policy_flatten")(x)

        # softmax activation converts vector to a vector of categorical probabilities
        # the elements of the output vector are in range(0,1) and sum to 1
        policy_out = Dense(4096,
                           kernel_regularizer=l2(self.__config.l2_reg),
                           activation="softmax", name="policy_out")(x)

        # ---------------------------------------------------------------------#
        # -------------------------     value head    -------------------------#

        x = Conv2D(filters=4,
                   kernel_size=1,
                   data_format="channels_last",
                   use_bias=False,
                   kernel_regularizer=l2(self.__config.l2_reg),
                   name="value_conv-1-4")(res_out)

        x = BatchNormalization(axis=-1,
                               name="value_batchnorm")(x)

        x = Activation("relu",
                       name="value_relu")(x)

        x = Flatten(name="value_flatten")(x)

        x = Dense(self.__config.value_fc_size,
                  kernel_regularizer=l2(self.__config.l2_reg),
                  activation="relu",
                  name="value_dense")(x)

        value_out = Dense(1,
                          kernel_regularizer=l2(self.__config.l2_reg),
                          activation="tanh",
                          name="value_out")(x)

        # ---------------------------------------------------------------------#
        # -------------------------     build model    ------------------------#

        # build the model with the above prepared layers
        self.model = Model(in_x, [value_out, policy_out], name=self.__config.modelname)

        # compile the model
        self.model.compile(
            # optimizer=tf.compat.v1.train.MomentumOptimizer(self.config.learning_rate_schedule, self.config.momentum,
            #                                               use_locking=False, name='Momentum', use_nesterov=False),
            # optimizer=SGD(lr=self.config.learning_rate, momentum=self.config.momentum),
            optimizer=Adam(learning_rate=0.5),
            loss={'value_out': self.__config.loss_funcs[0], 'policy_out': self.__config.loss_funcs[1]},
            loss_weights={'value_out': self.__config.loss_weights[0], 'policy_out': self.__config.loss_weights[1]})

        return self.model

    def __add_residual_layer(self, x: object, index: int) -> object:
        """Adds residual layer.

        Args:
          x:
            Object representing the input for the residual layer.
          index:
            Index of the residual layer

        Returns:
          Input with residual layer
        """
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=self.__config.cnn_filter_num,
                   kernel_size=self.__config.cnn_filter_size,
                   padding="same",
                   data_format="channels_last",
                   use_bias=False,
                   kernel_regularizer=l2(self.__config.l2_reg),
                   name=res_name + "_conv1-" + str(self.__config.cnn_filter_size) + "-" + str(
                       self.__config.cnn_filter_num))(x)

        x = BatchNormalization(axis=-1,
                               name=res_name + "_batchnorm1")(x)

        x = Conv2D(filters=self.__config.cnn_filter_num,
                   kernel_size=self.__config.cnn_filter_size,
                   padding="same",
                   data_format="channels_last",
                   use_bias=False,
                   kernel_regularizer=l2(self.__config.l2_reg),
                   name=res_name + "_conv2-" + str(self.__config.cnn_filter_size) + "-" + str(
                       self.__config.cnn_filter_num))(x)

        x = BatchNormalization(axis=-1,
                               name=res_name + "_batchnorm2")(x)

        # skip connections
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu",
                       name=res_name + "_relu2")(x)
        return x

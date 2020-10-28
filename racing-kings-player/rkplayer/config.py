class AlphaZeroConfig(object):
    """
        this is a config class that contains some parameters to control SELF PLAY simulations and NN training
        THIS PARAMETERS SHOULD BE CHANGED HERE LOCALLY !
    """

    def __init__(self):
        """
            Method to initialize all the needed parameters
        """
        # ---------------------------------------------------------------------#
        # ---------------------------     Selfplay    -------------------------#

        self.timout_poll_new_model = 100
        self.timout_poll_new_game_pool = 10

        # TensorRT configs
        self.use_tensorRT = True
        self.max_workspace_size_bytes = 14  # max allocated gb of gpu
        self.precision_mode = "FP16"  # precision mode (32FP/16FP/8INT)
        self.minimum_segment_size = 3  # min number of nodes in one tensorrt engine

        self.max_processes = 12  # max number of parallel raunning self-play processes
        self.num_pipes = 12  # max number of pipes (one per process)
        self.timeout_listen = 0.001

        # ---------------------------------------------------------------------#
        # ----------------------------     MCTS     ---------------------------#

        self.num_sampling_moves = 30  # number of steps (starting from root sate) where we choose moves/actions stochastically to enhance exploration
        self.max_moves = 512  # upper bound for game history/moves
        self.num_simulations = 120  # number of MCTS simulations per move

        # Pit configs for NNs evaluation
        self.pit_game_num = 4

        # UCB formula
        self.pb_c_base = 19652  # update of exploration rate
        self.pb_c_init = 1.25  # initial cput exploration rate

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # noise rate
        self.root_exploration_fraction = 0.25  # fraction for noise

        # ---------------------------------------------------------------------#
        # -------------------------     Neural Network    ---------------------#

        # Setups
        self.cnn_filter_num = 256  # number of filters in CNN layers
        self.cnn_first_filter_size = 5  # CNN filter size for the NN body
        self.cnn_filter_size = 3  # CNN filter size for policy and value heads
        self.res_layer_num = 20  # number of residual layers
        self.l2_reg = 1e-4  # weight of L2 regularization
        self.value_fc_size = 256  # size of fully connected layer for Value head

        # Training
        self.loss_funcs = ['mean_squared_error',
                           'categorical_crossentropy']  # type of loss functions: [value head, policy head]
        self.loss_weights = [0.5, 0.5]  # loss weights: [value head, policy head]
        # training data
        self.epochs = 100  # number of epochs to be ran over a training batch
        self.batch_size = 250  # batch size
        self.batch_mode = "normal"  # 'normal' vs 'finished_games'
        self.sample_mode = "uniform"  # 'normal' vs 'uniform' vs 'prio_short'

        # training steps
        self.training_steps = 20  # number of training steps
        self.window_size = 500  # size of data windows to sample training batchs

        # -------------------    general parameters    ------------------------#
        self.modelname = "rk_model_1.0_" + "{}-conv-{}-res-{}-fil_size".format(1, self.res_layer_num,
                                                                               self.cnn_filter_num)


def custom_selfplay_plan(gen):
    if gen <= 3:
        lower_than = 4
        num_rand_moves = 100
    elif gen <= 6:
        lower_than = 3
        num_rand_moves = 100
    elif gen <= 9:
        lower_than = 2
        num_rand_moves = 150
    elif gen <= 13:
        lower_than = 2
        num_rand_moves = 120
    elif gen <= 18:
        lower_than = 1
        num_rand_moves = 160
    elif gen <= 25:
        lower_than = 1
        num_rand_moves = 130
    elif gen <= 32:
        lower_than = 1
        num_rand_moves = 100
    elif gen <= 39:
        lower_than = 1
        num_rand_moves = 70
    elif gen <= 46:
        lower_than = 1
        num_rand_moves = 40
    else:
        lower_than = 1
        num_rand_moves = 0
    return lower_than, num_rand_moves

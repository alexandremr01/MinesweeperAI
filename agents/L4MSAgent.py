import random
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, activations, losses
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, concatenate, Flatten, Dense, ReLU
from tensorflow.keras.models import Model
from math import inf
from minesweeper import MinesweeperCore
from tensorflow.keras.optimizers import SGD


class L4MSAgent:
    """
    Represents a 6-Layer MineSweeper Agent.
    """
    def __init__(self, side, learning_rate=0.001):
        """
        Initializes the minesweeper agent.

        :param side: length of one side of the minesweeeper board.
        :type side: int.
        :param learning_rate: learning rate for the network.
        :type learning_rate: float.
        """
        self.side = side
        self.learning_rate = learning_rate
        self.model = self.make_model()
        self.num_incorretas = 0                        # this is only used of research purpose
        self.num_plays = 0

    def make_model(self):
        """
        Makes and returns the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """
        input_data = Input(shape=(self.side, self.side, 1))
        # Layer 1
        layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', name='conv_1', use_bias=False)(input_data)
        layer = BatchNormalization(name='norm_1')(layer)
        layer = ReLU()(layer)

        # Layer 2
        layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', name='conv_2', use_bias=False)(layer)
        layer = BatchNormalization(name='norm_2')(layer)
        layer = ReLU()(layer)

        # Layer 2
        layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', name='conv_3', use_bias=False)(layer)
        layer = BatchNormalization(name='norm_3')(layer)
        layer = ReLU()(layer)

        # Layer 4
        layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                       padding='same', name='conv_4', use_bias=False)(layer)
        layer = BatchNormalization(name='norm_4')(layer)
        layer = ReLU()(layer)

        layer = Flatten()(layer)

        layer = Dense(128, activation='relu')(layer)

        layer = Dense(self.side*self.side, activation='softmax')(layer)

        model = Model(inputs=input_data, outputs=layer, name='Minesweeper_Agent')
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        model.summary()
        return model

    def adjust_shape(self, table):
        """
        Adjusts the shape of the game board from a matrix to a full long vector.

        :param table: mine sweeper game board.
        :type table: Numpy matrix.
        :return: the newly shaped game board to a vector.
        :rtype: Numpy array.
        """
        table = (table+1)/10.0
        table = np.expand_dims(table, axis=0)
        table = np.expand_dims(table, axis=0)
        table = table.reshape(1, self.side, self.side, 1)
        return table

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        input_state = self.adjust_shape(state)
        output = self.model.predict(input_state)
        #print(output)
        self.num_plays = self.num_plays + 1
        while True:
            index = np.argmax(output)
            i = index // self.side
            j = index % self.side
            if state[i, j] != MinesweeperCore.UNKNOWN_CELL:
                self.num_incorretas = self.num_incorretas + 1
                output[0, index] = -inf
            else:
                break
        return i, j

    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name)


import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers, activations, losses
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from math import inf
from minesweeper import MinesweeperCore

class DQNAgent:
    """
    Represents a Deep Q-Networks (DQN) agent  that predicts Q(S,A) for every A.
    """
    def __init__(self, side, gamma=1.0, epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.98, learning_rate=0.001, buffer_size=4098):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param gamma: discount factor.
        :type gamma: float.
        :param epsilon: epsilon used in epsilon-greedy policy.
        :type epsilon: float.
        :param epsilon_min: minimum epsilon used in epsilon-greedy policy.
        :type epsilon_min: float.
        :param epsilon_decay: decay of epsilon per episode.
        :type epsilon_decay: float.
        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int.
        """
        self.side = side
        self.replay_buffer = deque(maxlen=buffer_size)  # giving a maximum length makes this buffer forget old memories
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.make_model()

    def get_cost_function(self):
        def cost_function(y_true, y_pred):
            return np.square(np.subtract(Y_true,Y_pred)).mean() 

        return cost_function

    def make_model(self):
        """
        Makes the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """
        input_data = Input(shape=(self.side, self.side, 1))
        # Layer 1
        layer = Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_data)
        layer = BatchNormalization(name='norm_1')(layer)
        layer = LeakyReLU(alpha=0.1, name='leaky_relu_1')(layer)

        # Layer 2
        layer = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(layer)
        layer = BatchNormalization(name='norm_2')(layer)
        layer = LeakyReLU(alpha=0.1, name='leaky_relu_2')(layer)

        layer = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_3', activation='linear')(layer)

        model = Model(inputs=input_data, outputs=layer, name='Minesweeper_Agent')
        model.compile(loss=losses.mse,
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def adjust_shape(self, table):
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
        nn_state = self.adjust_shape(state)
        qsa = self.model.predict(nn_state)
        qsa = qsa.reshape(self.side, self.side)
        filtered_qsa = np.multiply((state == MinesweeperCore.UNKNOWN_CELL), qsa)
        rand_val = random.uniform(0, 1)
        if rand_val > self.epsilon:
            i, j = np.unravel_index(np.argmax(filtered_qsa[:, :], axis=None), filtered_qsa[:, :].shape)
            return i, j
        else:
            while True:
                i = random.randint(0, self.side-1)
                j = random.randint(0, self.side-1)
                if (state[i, j]==MinesweeperCore.UNKNOWN_CELL):
                    return i, j

        #Boltzman
        # filtered_qsa[filtered_qsa==0] = -inf
        # T = 1
        # #print(state)
        # soft_qsa = np.exp(filtered_qsa/T)/np.sum(np.exp(filtered_qsa))
        # rand_val = random.uniform(0, 1)
        # #print(soft_qsa)
        # prob_sum = 0
        # #print(np.unravel_index(np.argmax(soft_qsa[:, :], axis=None), soft_qsa[:, :].shape))
        # action = np.unravel_index(np.argmax(soft_qsa[:, :], axis=None), soft_qsa[:, :].shape)
        # for i in range(0, soft_qsa.shape[0]):
        #     for j in range(0,soft_qsa.shape[1]):
        #         prob_sum += soft_qsa[i, j]
        #         if rand_val <= prob_sum:
        #             print(i, j)
        #             return i, j
        
        #print(action)
        #input("")
        #return action[0], action[1]
        

    def append_experience(self, state, action, reward, next_state, done):
        """
        Appends a new experience to the replay buffer (and forget an old one if the buffer is full).

        :param state: state.
        :type state: NumPy array with dimension (1, 2).
        :param action: action.
        :type action: int.
        :param reward: reward.
        :type reward: float.
        :param next_state: next state.
        :type next_state: NumPy array with dimension (1, 2).
        :param done: if the simulation is over after this experience.
        :type done: bool.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            state = self.adjust_shape(state)
            next_state = self.adjust_shape(next_state)
            target = self.model.predict(state)
            if not done:
                target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            else:
                target[0][action] = reward
            # Filtering out states and targets for training
            states.append(state[0])
            targets.append(target[0])
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss


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

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

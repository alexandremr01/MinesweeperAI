from minesweeper import MinesweeperCore
from minesweeper_environment import MinesweeperEnvironment
from agents.csp import MinesweeperAgent
import numpy as np
import os

# This file creates the dataset used in supervised learning.
# The Dataset consists of mappings of X to Y where X is the input of game states
# and Y is the output expected position to play on the minesweeper board for each game state.

# Dataset configuration
batch_size = 5000
dataset_size = 5000000
folder_name = 'dataset'

# Game/Actor Configuration
side = 8
bombs = 10
actor = MinesweeperAgent(side, bombs)
minesweeper = MinesweeperEnvironment(side, side, bombs)

batch = len([name for name in os.listdir(folder_name) if
             os.path.isfile(os.path.join(folder_name, name))]) // 2  # number of already existent dataset files

# Dataset initialization.
X = np.array([minesweeper.get_state()])
Y = np.array(np.zeros((1, side * side)))

for k in range(1, dataset_size + 1):
    state = minesweeper.get_state()
    X = np.concatenate([X, np.copy(state).reshape(1, side, side)])
    i, j = actor.act(state)
    next_state, reward, done = minesweeper.step(i, j)
    vector = np.zeros((1, side * side))
    vector[0, i * side + j] = 1
    Y = np.concatenate([Y, vector])
    if k % 1000 == 0:
        print('%s thousand done' % int(k / 1000))
    if k % batch_size == 0:
        X = np.expand_dims(X, axis=0).reshape(-1, side, side, 1)
        np.save(folder_name + '/dataset_X_' + str(batch), X[1:])
        np.save(folder_name + '/dataset_Y_' + str(batch), Y[1:])
        X = np.array([minesweeper.get_state()])
        Y = np.array(np.zeros((1, side * side)))
        batch += 1
    if done:
        minesweeper.reset()
        actor.reset()

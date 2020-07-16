from agents.L4MSAgent import L4MSAgent 
from minesweeper import MinesweeperCore
from minesweeper_environment import MinesweeperEnvironment
from tensorflow.keras.callbacks import TensorBoard
from agents.csp import MinesweeperAgent

import numpy as np
import os
from time import time


def adjust_dimension(table, side):
    m = table.shape[0]
    print(m)
    table = np.expand_dims(table, axis=0)
    table = table.reshape(m, side, side, 1)
    return table

def sample_trajectories(n, game, actor):
    game.reset()
    m = game.width
    X = np.array([game.get_state()])
    Y = np.array(np.zeros((m*m,1)))
    for k in range(0, n):
        state = game.get_state()
        X = np.concatenate([X, np.copy(game.get_state()).reshape(1, state.shape[0], state.shape[1]) ])
        print(game.get_state())
        i, j = actor.act(state)
        next_state, reward, done = game.step(i, j)
        vector = np.zeros((m*m, 1))
        vector[i*m + j] = 1
        Y = np.concatenate([Y, vector], axis=1)
        if done:
            game.reset()
            actor.reset()
    return X[1:], Y[:, 1:]

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

agent_name = 'l4ms01'
side = 8
bombs = 10
agent = L4MSAgent(side, bombs)
teacher = MinesweeperAgent(side, bombs)
minesweeper = MinesweeperEnvironment(side, side, bombs)

fig_format = 'png'
weights_filename = 'weights_'+agent_name+'.h5'
if os.path.exists(weights_filename):
    print('Loading weights from previous learning session.')
    agent.load(weights_filename)
else:
    print('No weights found from previous learning session.')


done = False
batch_size = 32 
return_history = []

tensorboard = TensorBoard(log_dir="logs\\{}".format(time()))

NUM_EPISODES = 1000

dataset_X, dataset_Y = sample_trajectories(1000, minesweeper, teacher)
for episodes in range(1, NUM_EPISODES + 1):
    new_X, new_Y = sample_trajectories(1000, minesweeper, teacher)
    dataset_X = np.concatenate([dataset_X, new_X], axis=0)
    dataset_Y = np.concatenate([dataset_Y, new_Y], axis=1)
    print(dataset_X.shape)
    nn_input = adjust_dimension(dataset_X, side)
    nn_output = dataset_Y.reshape(-1, side*side)
    agent.model.fit(nn_input, nn_output, epochs = 30, callbacks=[tensorboard], shuffle = True, batch_size = batch_size)
    agent.save(weights_filename)
    print(len(dataset_X))
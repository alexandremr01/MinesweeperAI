from agents.L4MSAgent import L4MSAgent 
from minesweeper import MinesweeperCore
from minesweeper_environment import MinesweeperEnvironment
from tensorflow.keras.callbacks import TensorBoard
from agents.csp import MinesweeperAgent

import numpy as np
import os
from time import time

def sample_trajectories(n, game, actor):
    game.reset()
    X = np.array([game.get_state()])
    Y = np.array([(0, 0)])
    for k in range(0, n):
        state = game.get_state()
        X = np.concatenate([X, np.copy(game.get_state()).reshape(1, state.shape[0], state.shape[1]) ])
        i, j = actor.act(state)
        next_state, reward, done = game.step(i, j)
        Y = np.concatenate([Y, [(i,j)]])
        if done:
            game.reset()
            actor.reset()
    return X[1:], Y[1:]

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

NUM_EPISODES = 5

dataset_X, dataset_Y = sample_trajectories(1000, minesweeper, teacher)
for episodes in range(1, NUM_EPISODES + 1):
    new_X, new_Y = sample_trajectories(1000, minesweeper, teacher)
    dataset_X = np.concatenate([dataset_X, new_X])
    dataset_Y = np.concatenate([dataset_Y, new_Y])
    agent.model.fit(dataset_X, dataset_Y, epochs = 20, callbacks=[tensorboard], shuffle = True, batch_size = batch_size)
    agent.save(weights_filename)
    print(len(dataset_X))
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
    #X = np.load('dataset_X.npy')
    #Y = np.load('dataset_Y.npy')
    for k in range(len(X), n):
        state = game.get_state()
        X = np.concatenate([X, np.copy(game.get_state()).reshape(1, state.shape[0], state.shape[1]) ])
        #print(game.get_state())
        #print("Salvos: ")
        #print(X)
        i, j = actor.act(state)
        next_state, reward, done = game.step(i, j)
        vector = np.zeros((m*m, 1))
        vector[i*m + j] = 1
        Y = np.concatenate([Y, vector], axis=1)
        if k % 1000 == 0:
          print('%s thousand done'%(k/1000))
          np.save('dataset_X', X[1:])
          np.save('dataset_Y', Y[:, 1:])
        if done:
            game.reset()
            actor.reset()
        
    return X[1:], Y[:, 1:]

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

agent_name = 'l4ms01'
side = 8
bombs = 10
#agent = L4MSAgent(side, bombs)
teacher = MinesweeperAgent(side, bombs)
minesweeper = MinesweeperEnvironment(side, side, bombs)

# fig_format = 'png'
# weights_filename = 'weights_'+agent_name+'.h5'
# if os.path.exists(weights_filename):
#     print('Loading weights from previous learning session.')
#     agent.load(weights_filename)
# else:
#     print('No weights found from previous learning session.')


done = False
batch_size = 32 
return_history = []

NUM_EPISODES = 5
steps_per_episode = 1000

dataset_X, dataset_Y = sample_trajectories(1000000, minesweeper, teacher)

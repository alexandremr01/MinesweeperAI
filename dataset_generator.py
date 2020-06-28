import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os


def random_actor(state):
  size = state.shape[0]
  while True:
    x = random.randint(0, size-1)
    y = random.randint(0, size-1)
    if state[x, y] == MinesweeperCore.UNKNOWN_CELL:
      break
  return x, y

# This scripts runs a random policy to generate data.
# It runs the policy untill completing the board.

size = 8
NUM_EPISODES = 2
bombs=10
minesweeper = MinesweeperEnvironment(size, size, bombs, 1)

victories = 0
plays_to_die = []
open_percentage = []

with open('dataset_boards_1', 'w') as board_dataset , open('dataset_bombs_1','w') as bomb_dataset:
    for episodes in range(1, NUM_EPISODES + 1):
        state = minesweeper.reset()
        while minesweeper.is_finished() != True:
            action = random_actor(minesweeper.get_state(xray=True)) 
            #print(action)
            next_state, reward, done = minesweeper.step(action[0], action[1])
            state = next_state
            #minesweeper.print_board()
            #input("")
            bomb_dataset.write(str(minesweeper.game.bomb_positions)+'\n')
            board_dataset.write(str(minesweeper.get_state(xray=True).reshape(1, -1))+'\n')
        print('Played ',episodes,'/',NUM_EPISODES)
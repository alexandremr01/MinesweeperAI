import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
import matplotlib.pyplot as plt
import os
from csp import MinesweeperAgent



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
bombs = 10
win_threshold = 1.0
game = MinesweeperEnvironment(size, size, bombs, win_threshold)
agent = MinesweeperAgent(size,bombs)

NUM_EPISODES = 1000

victories = 0
plays_to_die = []
open_percentage = []

boards = []
actions = []

for episodes in range(1, NUM_EPISODES + 1):
  state = game.reset()
  agent.reset()
  n=0
  while game.is_finished() != True:
    action = agent.act(game.get_state())
    next_position = game.step(action[0], action[1])
    n +=1
    if (n>30):
      game.print_board()
      print(action)
    #boards.append(next_position)
   # actions.append(action)
  print('Played ',episodes,'/',NUM_EPISODES)

print("dataset size: " + str(len(actions)))
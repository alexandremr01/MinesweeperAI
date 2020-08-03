import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
#from agents.csp import MinesweeperAgent
import matplotlib.pyplot as plt
import os
from agents.L4MSAgent import L4MSAgent

# This script runs an actor and evaluate it.
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def random_actor(state):
  size = state.shape[0]
  while True:
    x = random.randint(0, size-1)
    y = random.randint(0, size-1)
    if state[x, y] == MinesweeperCore.UNKNOWN_CELL:
      break
  return x, y

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


size = 8
NUM_EPISODES = 1000
bombs=10
game = MinesweeperEnvironment(size, size, bombs)

victories = 0
plays_to_die = []
open_percentage = []

#agent = DQNAgent(size)
#agent = MinesweeperAgent(size,bombs)
agent = L4MSAgent(size, bombs)
model = 'best_model.hdf5'

if os.path.exists(model):
    print('Loading weights from previous learning session.')
    agent.load(model)
else:
    print('No weights found from previous learning session.')

for episodes in range(1, NUM_EPISODES + 1):
    state = game.reset()
    #agent.reset()
    plays = 0
    while game.is_finished() != True:
        #action = random_actor(game.get_state()) # Use this to test random policy
        action = agent.act(game.get_state())
        #print(action)
        next_state = game.step(action[0], action[1])
        #game.print_board()
        state = next_state
        plays += 1
    if game.is_victory():
      victories += 1
    else:
      plays_to_die.append(plays-1)
    open_percentage.append(game.get_open_percentage())
    if episodes % 100 == 0:
      print('Played ',episodes,'/',NUM_EPISODES)

# Prints mean return
print('Mean return: ', np.mean(open_percentage), ' +/- ', np.std(open_percentage))
print('Mean plays to die: ', np.mean(plays_to_die), ' +/- ', np.std(plays_to_die))
print('Victory percentage: ', victories/NUM_EPISODES)

# Plots return history
plt.hist(plays_to_die, bins=list(range(0,np.max(plays_to_die))))
plt.xlabel('# plays')
plt.ylabel('# episodes')
plt.title('Histogram of number of plays untill defeat')
plt.show()

plt.hist(open_percentage)
plt.xlabel('% open')
plt.ylabel('# episodes')
plt.title('Histogram of open percentage (>0.8 is win)')
plt.show()

import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os

# This script runs an actor and evaluate it.

def random_actor(state):
  size = state.shape[0]
  while True:
    x = random.randint(0, size-1)
    y = random.randint(0, size-1)
    if state[x, y] == MinesweeperCore.UNKNOWN_CELL:
      break
  return x, y


size = 8
NUM_EPISODES = 1000
bombs=10
game = MinesweeperEnvironment(size, size, bombs)

victories = 0
plays_to_die = []
open_percentage = []
agent = DQNAgent(size)

if os.path.exists('minesweeper.h5'):
    print('Loading weights from previous learning session.')
    agent.load("minesweeper.h5")
else:
    print('No weights found from previous learning session.')

for episodes in range(1, NUM_EPISODES + 1):
    state = game.reset()
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
    print('Played ',episodes,'/',NUM_EPISODES)

#print(victories)
#print(plays_to_die)
#print(open_percentage)
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
print(plays_to_die)

plt.hist(open_percentage)
plt.xlabel('% open')
plt.ylabel('# episodes')
plt.title('Histogram of open percentage (>0.8 is win)')
plt.show()
print(open_percentage)
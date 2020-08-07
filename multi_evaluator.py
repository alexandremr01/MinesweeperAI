import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
from agents.csp import MinesweeperAgent
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

# Set game configurations. CSP can play in any board size. The remaining agents can only play in 8x8 board.
size = 8
NUM_EPISODES = 100
bombs = [8, 10, 12]
# Choose an agent
agent_name = 'h_csp' # h_csp (heuristic csp), nh_csp (non-heuristic csp) or l4ms

##
if agent_name == 'l4ms':
    agent = L4MSAgent(size)
    weights = 'results/best_model.hdf5'
    if os.path.exists(weights):
        print('Loading weights from previous learning session.')
        agent.load(weights)
    else:
        print('No weights found from previous learning session.')

victories = 0
plays_to_die = []
open_percentage = []
num_games = 3 # Do not change this line
games_open_percentage = []
games_plays_to_die = []
games_victory_percentage = []
for current_game in range(num_games):
    game = MinesweeperEnvironment(size, size, bombs[current_game])
    if agent_name == 'h_csp':
        heuristic = True
        agent = MinesweeperAgent(size, bombs[current_game], heuristic)
    elif agent_name == 'nh_csp':
        heuristic = False
        agent = MinesweeperAgent(size, bombs[current_game], heuristic)
    victories = 0
    plays_to_die = []
    open_percentage = []
    for episodes in range(1, NUM_EPISODES + 1):
        state = game.reset()
        agent.reset()
        plays = 0
        while game.is_finished() != True:
            #action = random_actor(game.get_state()) # Use this to test random policy
            action = agent.act(game.get_state())
            next_state = game.step(action[0], action[1])
            state = next_state
            plays += 1
        if game.is_victory():
          victories += 1
        else:
          plays_to_die.append(plays-1)
        open_percentage.append(game.get_open_percentage()*100)
        print('Game', current_game + 1, '/', num_games, 'Played',episodes,'/',NUM_EPISODES, 'Open percentage:', game.get_open_percentage()*100, '%')
    games_open_percentage.append(open_percentage)
    games_plays_to_die.append(plays_to_die)
    victory_percentage = victories / NUM_EPISODES
    games_victory_percentage.append(victory_percentage*100)

print('Win rate:\n', bombs[0], 'bombs -', games_victory_percentage[0], '%\n', bombs[1], 'bombs -', games_victory_percentage[1], '%\n', bombs[2]
, 'bombs -', games_victory_percentage[2], '%')

# Plots return history
plt.hist(games_plays_to_die[0], bins=list(range(0,np.max(np.max(games_plays_to_die)))), label='8 Bombs', alpha=0.6, color='b')
plt.hist(games_plays_to_die[1], bins=list(range(0,np.max(np.max(games_plays_to_die)))), label='10 Bombs', alpha=0.6, color='darkgreen')
plt.hist(games_plays_to_die[2], bins=list(range(0,np.max(np.max(games_plays_to_die)))), label='12 Bombs', alpha=0.6, color='r')
plt.legend(loc='upper right')
plt.xlabel('# plays')
plt.ylabel('# episodes')
plt.title('Histogram of number of plays untill defeat')
plt.show()

plt.hist(games_open_percentage[0], bins=20, label='8 Bombs', alpha=0.6, color='b')
plt.hist(games_open_percentage[1], bins=20, label='10 Bombs', alpha=0.6, color='darkgreen')
plt.hist(games_open_percentage[2], bins=20, label='12 Bombs', alpha=0.6, color='r')
plt.legend(loc='upper left')
plt.xlabel('% open')
plt.ylabel('# episodes')
plt.title('Histogram of open percentage')
plt.show()

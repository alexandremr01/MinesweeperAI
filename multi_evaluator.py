import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
from agents.csp import MinesweeperAgent
import matplotlib.pyplot as plt
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
NUM_EPISODES = 100
bombs = [8, 12, 16]

victories = 0
plays_to_die = []
open_percentage = []

if os.path.exists('minesweeper.h5'):
    print('Loading weights from previous learning session.')
    agent.load("minesweeper.h5")
else:
    print('No weights found from previous learning session.')

num_games = 3
games_open_percentage = []
games_plays_to_die = []
games_victory_percentage = []
for current_game in range(num_games):
    game = MinesweeperEnvironment(size, size, bombs[current_game])
    agent = MinesweeperAgent(size, bombs[current_game])
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
            #print(action)
            next_state = game.step(action[0], action[1])
            #game.print_board(True)
            state = next_state
            plays += 1
        if game.is_victory():
          victories += 1
        else:
          plays_to_die.append(plays-1)
        open_percentage.append(game.get_open_percentage())
        print('Game', current_game + 1, '/', num_games, 'Played',episodes,'/',NUM_EPISODES, "Open percentage:", game.get_open_percentage())
    games_open_percentage.append(open_percentage)
    games_plays_to_die.append(plays_to_die)
    victory_percentage = victories / NUM_EPISODES
    games_victory_percentage.append(victory_percentage)

print('Win rate:', bombs[0], 'bombs -', games_victory_percentage[0], '//', bombs[1], 'bombs -', games_victory_percentage[1], '//', bombs[2]
, 'bombs -', games_victory_percentage[2])

# Plots return history
plt.hist(games_plays_to_die[0], bins=list(range(0,np.max(np.max(games_plays_to_die)))), label='8 Bombs', alpha=0.6, color='b')
plt.hist(games_plays_to_die[1], bins=list(range(0,np.max(np.max(games_plays_to_die)))), label='12 Bombs', alpha=0.6, color='darkgreen')
plt.hist(games_plays_to_die[2], bins=list(range(0,np.max(np.max(games_plays_to_die)))), label='16 Bombs', alpha=0.6, color='r')
plt.legend(loc='upper right')
plt.xlabel('# plays')
plt.ylabel('# episodes')
plt.title('Histogram of number of plays untill defeat')
plt.show()

plt.hist(games_open_percentage[0], label='8 Bombs', alpha=0.6, color='b')
plt.hist(games_open_percentage[1], label='12 Bombs', alpha=0.6, color='darkgreen')
plt.hist(games_open_percentage[2], label='16 Bombs', alpha=0.6, color='r')
plt.legend(loc='upper left')
plt.xlabel('% open')
plt.ylabel('# episodes')
plt.title('Histogram of open percentage')
plt.show()

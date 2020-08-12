import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
from agents.csp import MinesweeperAgent
import matplotlib.pyplot as plt
import os
from agents.L4MSAgent import L4MSAgent
# This script runs an actor and evaluate it.

# Some PCs needed these configurations in order to run with GPU.
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#  Comment this line to enable running using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
NUM_EPISODES = 10
bombs = [8, 10, 12]
# Choose an agent
agent_name = 'l4ms' # h_csp (heuristic csp), nh_csp (non-heuristic csp) or l4ms

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
games_wrong_plays = []
games_victory_percentage = []
game_guesses = [] # Just for h_csp or nh_csp
guesses = [] # Just for h_csp or nh_csp
for current_game in range(num_games):
    game = MinesweeperEnvironment(size, size, bombs[current_game])
    if agent_name == 'h_csp':
        heuristic = True
        agent = MinesweeperAgent(size, bombs[current_game], heuristic)
    elif agent_name == 'nh_csp':
        heuristic = False
        agent = MinesweeperAgent(size, bombs[current_game], heuristic)
    victories = 0
    if agent_name == 'l4ms':
        agent.num_incorretas = 0
        agent.num_plays = 0
    plays_to_die = []
    open_percentage = []
    guesses = []
    for episodes in range(1, NUM_EPISODES + 1):
        state = game.reset()
        agent.reset()
        plays = 0
        guess_percentage = 0
        while game.is_finished() != True:
            #action = random_actor(game.get_state()) # Use this to test random policy
            action = agent.act(game.get_state())
            next_state = game.step(action[0], action[1])
            if agent_name == 'h_csp' or agent_name == 'nh_csp':
                if agent.guess_flag == True:
                    guess_percentage = game.get_open_percentage()*100
            state = next_state
            plays += 1
        if game.is_victory():
          victories += 1
          if agent_name == 'h_csp' or agent_name == 'nh_csp':
              guesses.append(guess_percentage)
        else:
          plays_to_die.append(plays-1)
        open_percentage.append(game.get_open_percentage()*100)
        print('Game', current_game + 1, '/', num_games, 'Played',episodes,'/',NUM_EPISODES, 'Open percentage:', game.get_open_percentage()*100, '%')
    game_guesses.append(guesses)
    games_open_percentage.append(open_percentage)
    games_plays_to_die.append(plays_to_die)
    victory_percentage = victories / NUM_EPISODES
    games_victory_percentage.append(victory_percentage*100)
    if agent_name == 'l4ms':
        wrong_percentage = agent.num_incorretas / agent.num_plays
        games_wrong_plays.append(wrong_percentage*100)

print('Win rate:\n', bombs[0], 'bombs -', games_victory_percentage[0], '%\n', bombs[1], 'bombs -', games_victory_percentage[1], '%\n', bombs[2]
, 'bombs -', games_victory_percentage[2], '%')

if agent_name == 'l4ms':
    print('Wrong plays rate:', bombs[0], 'bombs -', games_wrong_plays[0], '%//', bombs[1], 'bombs -', games_wrong_plays[1], '%//', bombs[2]
    , 'bombs -', games_wrong_plays[2], '%')


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

if agent_name == 'h_csp' or agent_name == 'nh_csp':
    plt.hist(game_guesses[0], bins=20, label='8 Bombs', alpha=0.6, color='b')
    plt.hist(game_guesses[1], bins=20, label='10 Bombs', alpha=0.6, color='darkgreen')
    plt.hist(game_guesses[2], bins=20, label='12 Bombs', alpha=0.6, color='r')
    plt.legend(loc='upper right')
    plt.xlabel('% open ')
    plt.ylabel('# episodes')
    plt.title('Open board percentage after last guessed move')
    plt.show()

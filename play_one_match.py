import time
import numpy as np
import random
import os
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
from agents.csp import MinesweeperAgent
from agents.L4MSAgent import L4MSAgent

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set game configurations. CSP can play in any board size. The remaining agents can only play in 8x8 board.
size = 8
bombs = 10
game = MinesweeperEnvironment(size, size, bombs)

# Choose an agent
agent_name = 'h_csp' # h_csp (heuristic csp), nh_csp (non-heuristic csp) or l4ms

if agent_name == 'h_csp':
    heuristic = True
    agent = MinesweeperAgent(size, bombs, heuristic)
elif agent_name == 'nh_csp':
    heuristic = False
    agent = MinesweeperAgent(size, bombs, heuristic)
elif agent_name == 'l4ms':
    agent = L4MSAgent(size, bombs)
    model = 'results/best_model.hdf5'
    if os.path.exists(model):
        print('Loading weights from previous learning session.')
        agent.load(model)
    else:
        print('No weights found from previous learning session.')

while game.is_finished() != True:
    action = agent.act(game.get_state())
    next_state = game.step(action[0], action[1])
    game.print_board()
    print("Played", action)
    state = next_state
    time.sleep(1)

print("Open percentage:", game.get_open_percentage()*100, "%")
game.print_bomb_positions()

from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
from agents.csp import MinesweeperAgent
import numpy as np

size = 8
bombs = 10
win_threshold = 1.0
game = MinesweeperEnvironment(size, size, bombs, win_threshold)
agent = MinesweeperAgent(size, bombs)

while game.game.isPlaying():
    action = agent.act(game.get_state())
    next_position = game.step(action[0], action[1])
    #game.game.play(*position)
    print("Played", action)
    game.print_board()
    #board = game.game.get_board()
    #position = agent.act(board)

# Debugging "simplifies constraints" feature
#board1 = np.matrix([[3, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
#board2 = np.matrix([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 2, -1,], [-1, -1, -1, -1]])
#constraints, constraints_var = agent.read(board2)
#for constraint in constraints:
#    print(constraint.variables)
#    print(constraint.value)
#coupled_constraints = agent.coupled(constraints,constraints_var)
#for constraints in coupled_constraints:
#    print("---")
#    for constraint in constraints:
#        print(constraint.variables)
#        print(constraint.value)
#    print("---")
#ans = agent.solve(coupled_constraints)
#for sol in ans:
#    print(sol)
#agent.PlaySimpleStrategy(ans, constraints_var)

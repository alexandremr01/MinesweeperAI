from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
from csp import MinesweeperAgent
import numpy as np

size = 6
bombs = 5
win_threshold = 1.0
game = MinesweeperEnvironment(size, size, bombs, win_threshold)
agent = MinesweeperAgent()

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

next_position = (2, 2)
game.print_board()

while game.game.isPlaying():
    cell = input()
    row = int(cell[0])
    column = int(cell[2])
    next_position = (row, column)
    game.game.play(*next_position)
    print("Played", next_position)
    game.print_board()
    #board = game.game.get_board()
    #constraints, constrained_var = agent.read(board)
    #coupled_constraints = agent.coupled(constraints,constrained_var)
    #ans = agent.solve(coupled_constraints)
    #for sol in ans:
        #print(sol)
        #for constraint in constraints:
            #print(constraint.variables)
            #print(constraint.value)
    #next_position = agent.PlaySimpleStrategy(ans,constrained_var)

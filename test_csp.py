from csp import MinesweeperConstraint, MinesweeperCSP

constrained_variables = [(0,1), (0,2), (0,3)]
constraint_1 = MinesweeperConstraint(constrained_variables, 2)
constraint_2 = MinesweeperConstraint([(0,1), (0,3)], 1)
CSP = MinesweeperCSP(constrained_variables, {})
CSP.add_constraint(constraint_1)
CSP.add_constraint(constraint_2)
assignment = {}
answer = CSP.backtracking_search(assignment)
print(answer)

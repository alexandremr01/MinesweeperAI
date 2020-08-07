from agents.csp import MinesweeperConstraint, CSPSolver

constrained_variables = [(0, 1), (0, 2), (0, 3), (0, 4)]
constraint_1 = MinesweeperConstraint([(0, 1), (0, 2)], 1)
constraint_2 = MinesweeperConstraint([(0, 3), (0, 4)], 1)
CSP = CSPSolver(constrained_variables, {})
CSP.add_constraint(constraint_1)
CSP.add_constraint(constraint_2)
assignment = {}
CSP.backtracking_search(assignment)
answer = CSP.solutions.copy()
print(answer)

# Features
# Consegue resolver equações separadas mas que tem solucao !!
# Fornece todas as soluçoes para a equaçao dada !!

# Faltando
# Um conjunto de eqs com solucao + um conjunto de eq sem solucao -> sem solucao no metodo
# Simplificacao das constraints

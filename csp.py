class MinesweeperConstraint:
    """
    Base class for minesweeper constraints.
    """
    def __init__(self, variables, value):
        """
        Creates a constraint.

        :type variables: list of tuples (i, j).
        :type value: int.
        """
        self.variables = variables
        self.value = value

    def solvable(self, assignment):
        """
        Checks if a minesweeper constraint still solvable after assignment

        :param assignment: Maps each variable to a value
        :type assignment: dictionary
        :return type: bool
        """
        new_value = self.value
        non_constrained = 0
        for variable in self.variables:
            if variable not in assignment:
                non_constrained = non_constrained + 1
                continue
            new_value = new_value - assignment[variable]
        if new_value >= 0 and new_value <= non_constrained:
            return True
        else:
            return False

class CSPSolver:
    """
    Constraint-satisfaction problem solver for minesweeper.
    """
    def __init__(self, variables=[], constraints={}):
        """
        :type variables: list of tuples (i, j).
        :type domains: list.
        :type constraints: dictionary.
        :type solutions: list of dictionaries.
        """
        self.variables = variables
        self.domains = [0, 1]
        self.constraints = constraints
        self.solutions = []
        for variable in self.variables:
            self.constraints[variable] = []

    def add_variable(self, variable):
        """
        Adds a variable

        """
        self.variable.append(variable)
        self.constraints[variable] = []

    def add_constraint(self, constraint):
        """
        Adds a constraint

        :type constraint: MinesweeperConstraint
        """
        for variable in constraint.variables:
            if variable not in self.variables:
                raise LookupError("Variable not in CSP")
            else:
                self.constraints[variable].append(constraint)

    def consistent_assignment(self, variable, assignment):
        """
        Given a new set of assignments, checks if variable constraints
        still have a solution.

        :type variable: tuple
        :type assignment: dictionary
        """
        for constraint in self.constraints[variable]:
            if not constraint.solvable(assignment):
                return False
        return True

    def backtracking_search(self, assignment={}):
        """
        Gives all solutions for the CSP

        :param assignment: Maps a set of variables to a value
        :type assignment: dictionary
        """
        if len(assignment) == len(self.variables):
            self.solutions.append(assignment)
            return None
        unassigned = []
        for variable in self.variables:
            if variable not in assignment:
                unassigned.append(variable)
        first = unassigned[0]
        for value in self.domains:
            local_assignment = assignment.copy()
            local_assignment[first] = value
            if self.consistent_assignment(first, local_assignment):
                result = self.backtracking_search(local_assignment)
        return None

    def sol(self):
        """
        Returns the solutions

        """
        return self.solutions


class MinesweeperAgent:

    def __init__(self):
        self.constraints = []
        self.play = []

    def PlaySimpleStrategy(self, anwsers, variables):
        """
        Returns a position (i, j) to play using SS

        """
        probabilities = {} # variables as keys // elements: [all cases, cases of 0, probability to be 0]
        for variable in variables:
            probabilities[variable] = [0, 0, 0]
        for ans in anwsers:
            for solution in ans:
                for variable in solution:
                    probabilities[variable][0] = probabilities[variable][0] + 1
                    if(solution[variable] == 0):
                        probabilities[variable][1] = probabilities[variable][1] + 1
        best_position = [(0, 0), 0]
        for variable in variables:
            probabilities[variable][2] = probabilities[variable][1] / probabilities[variable][0]
            if(probabilities[variable][2] > best_position[1]):
                best_position[1] = probabilities[variable][2]
                best_position[0] = variable
        return best_position[0]

    def solve(self, coupled_constraints):
        """
        Solves coupled constraints

        """
        variables = []
        answer = []
        for constraint_set in coupled_constraints:
            variables = []
            for constraint in constraint_set:
                variables = variables + constraint.variables
            variables = set(variables)
            variables = list(variables)
            solver = CSPSolver(variables)
            for constraint in constraint_set:
                solver.add_constraint(constraint)
            solver.backtracking_search()
            solutions = solver.sol()
            answer.append(solutions)
        return answer

    def coupled(self, constraints, variables):
        """
        Returns all coupled sets of constraints.

        """
        coupled_constraints = []
        for variable in variables:
            list = []
            for constraint in constraints:
                if variable in constraint.variables:
                    list.append(constraint)
            coupled_constraints.append(list)
        return coupled_constraints

    def read(self, board):
        """
        Reads minesweeper board. Returns constraints and all variables constrained.

        :type board: numpy matrix
        """
        self.constraints = []
        constrained_var = set()
        height = board.shape[0]
        width = board.shape[1]
        variables = None
        value = None
        constraint = None
        # Falta
        # trocar -1 por UNKNOWN_CELL nao sei porque n foi
        # Corners
        if(board[0, 0] > 0):
            variables = []
            positions = [(0,1), (1,0), (1,1)]
            for variable in positions:
                if(board[variable] == -1):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[0, 0]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        if(board[0, width-1] > 0):
            variables = []
            positions = [(0,width-2), (1,width-1), (1,width-2)]
            for variable in positions:
                if(board[variable] == -1):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[0, width-1]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        if(board[height-1, 0] > 0):
            variables = []
            positions = [(height-2,0), (height-2,1), (height-1,1)]
            for variable in positions:
                if(board[variable] == -1):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[height-1, 0]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        if(board[height-1, width-1] > 0):
            variables = []
            positions = [(height-2,width-1), (height-2,width-2), (height-1,width-2)]
            for variable in positions:
                if(board[variable] == -1):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[height-1, width-1]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        # Edges
        for i in range(width - 2):
            if(board[0, i+1] > 0):
                variables = []
                positions = [(0, i), (0, i+2), (1, i), (1, i+1), (1, i+2)]
                for variable in positions:
                    if(board[variable] == -1):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[0, i+1]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
            if(board[height-1, i+1] > 0):
                variables = []
                positions = [(height-1, i), (height-1, i+2), (height-2, i), (height-2, i+1), (height-2, i+2)]
                for variable in positions:
                    if(board[variable] == -1):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[0, i+1]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
            if(board[i+1, 0] > 0):
                variables = []
                positions = [(i, 0), (i+2, 0), (i, 1), (i+1, 1), (i+2, 1)]
                for variable in positions:
                    if(board[variable] == -1):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[i+1, 0]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
            if(board[i+1, height-1] > 0):
                variables = []
                positions = [(i, height-1), (i+2, height-1), (i, height-2), (i+1, height-2), (i+2, height-2)]
                for variable in positions:
                    if(board[variable] == -1):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[i+1, height-1]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
        # Inside
        for i in range(width - 2):
            for j in range(height - 2):
                if(board[i+1, j+1] > 0):
                    variables = []
                    positions = [(i+1, j), (i+1, j+2), (i, j), (i, j+1), (i, j+2), (i+2, j), (i+2, j+1), (i+2, j+2)]
                    for variable in positions:
                        if(board[variable] == -1):
                            variables.append(variable)
                            constrained_var.add(variable)
                    value = board[i+1, j+1]
                    constraint = MinesweeperConstraint(variables, value)
                    self.constraints.append(constraint)
        # Simplifies constraints
        for i in range(len(self.constraints)):
            j = i + 1
            while(j < len(self.constraints)):
                if(set(self.constraints[i].variables).issubset(set(self.constraints[j].variables))):
                    variables = [var for var in self.constraints[j].variables if var not in self.constraints[i].variables]
                    value = self.constraints[j].value - self.constraints[i].value
                    self.constraints[j].variables = variables
                    self.constraints[j].value = value
                j = j + 1
        constraints = [constraint for constraint in self.constraints if constraint.variables != []]
        self.constraints = constraints
        return self.constraints, constrained_var

from minesweeper import MinesweeperCore
import numpy as np

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
        self.constraints = {}
        self.solutions = []
        for variable in self.variables:
            self.constraints[variable] = []
        for constraint in constraints:
            self.add_constraint(constraint)

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

    def get_answers(self):
        """
        Runs backtracking search and returns solutions.

        """
        self.backtracking_search()
        return self.solutions


class MinesweeperAgent:
    """
    Simple strategy Minesweeper agent.
    """
    def __init__(self, size, num_bombs):
        self.initial_position = (int(size/2), int(size/2))
        self.board_size = size
        self.num_bombs = num_bombs
        self.constraints = []
        self.nobomb_position = [self.initial_position]
        self.bomb_position = []
        self.unknown_position = []

    def reset(self):
        """
        Resets agent.

        """
        self.constraints = []
        self.nobomb_position = [self.initial_position]
        self.bomb_position = []
        self.unknown_position = []

    def act(self, board):
        """
        Returns a position (i, j) to play using simple strategy.

        :type board: numpy matrix.
        """
        if(len(self.nobomb_position) != 0):
            return self.nobomb_position.pop(0)
        constraints, constrained_variables = self.read_board(board)
        if(len(self.nobomb_position) != 0):
            return self.nobomb_position.pop(0)
        if(len(self.bomb_position) == self.num_bombs):
            for position in self.unknown_position:
                if(position not in self.bomb_position):
                    self.nobomb_position.append(position)
            return self.nobomb_position.pop(0)
        csp = CSPSolver(constrained_variables, constraints)
        answers = csp.get_answers()
        # Corner heuristic # # #
        corner_position = [(0, 0), (self.board_size - 1, self.board_size - 1), (0, self.board_size - 1), (self.board_size - 1, 0)]
        for position in corner_position:
            if(position not in self.bomb_position and position in self.unknown_position):
                return position
        # # # # # # # # # # # # #
        probabilities = {} # Variables as keys, [All cases, Cases of 0, Probability to be 0]
        for variable in constrained_variables:
            probabilities[variable] = [0, 0, 0]
        for ans in answers:
                for variable in ans:
                    probabilities[variable][0] = probabilities[variable][0] + 1
                    if(ans[variable] == 0):
                        probabilities[variable][1] = probabilities[variable][1] + 1
        best_position = [(0, 0), 0]
        for variable in constrained_variables:
            if(probabilities[variable][0] != 0):
                probabilities[variable][2] = probabilities[variable][1] / probabilities[variable][0]
            if(probabilities[variable][2] > best_position[1] and variable not in self.bomb_position):
                best_position[1] = probabilities[variable][2]
                best_position[0] = variable
        if(best_position[1] == 0):
            playable_positions = [position for position in self.unknown_position if position not in self.bomb_position]
            rand_index = np.random.randint(len(playable_positions))
            best_position[0] = playable_positions[rand_index]
        return best_position[0]

    def read_board(self, board):
        """
        Reads minesweeper board. Returns constraints and all variables constrained.

        :type board: numpy matrix
        """
        self.constraints = []
        self.unknown_position = []
        constrained_var = set()
        height = board.shape[0]
        width = board.shape[1]
        variables = None
        value = None
        constraint = None
        # Corner constraints
        if(board[0, 0] == MinesweeperCore.UNKNOWN_CELL):
            self.unknown_position.append((0, 0))
        elif(board[0, 0] > 0):
            variables = []
            positions = [(0,1), (1,0), (1,1)]
            for variable in positions:
                if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[0, 0]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        if(board[0, width-1] == MinesweeperCore.UNKNOWN_CELL):
            self.unknown_position.append((0, width-1))
        elif(board[0, width-1] > 0):
            variables = []
            positions = [(0,width-2), (1,width-1), (1,width-2)]
            for variable in positions:
                if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[0, width-1]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        if(board[height-1, 0] == MinesweeperCore.UNKNOWN_CELL):
            self.unknown_position.append((height-1, 0))
        elif(board[height-1, 0] > 0):
            variables = []
            positions = [(height-2,0), (height-2,1), (height-1,1)]
            for variable in positions:
                if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[height-1, 0]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        if(board[height-1,width-1] == MinesweeperCore.UNKNOWN_CELL):
            self.unknown_position.append((height-1, width-1))
        elif(board[height-1, width-1] > 0):
            variables = []
            positions = [(height-2,width-1), (height-2,width-2), (height-1,width-2)]
            for variable in positions:
                if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                    variables.append(variable)
                    constrained_var.add(variable)
            value = board[height-1, width-1]
            constraint = MinesweeperConstraint(variables, value)
            self.constraints.append(constraint)
        # Edges constraints
        for i in range(width - 2):
            if(board[0, i+1] == MinesweeperCore.UNKNOWN_CELL):
                self.unknown_position.append((0, i+1))
            elif(board[0, i+1] > 0):
                variables = []
                positions = [(0, i), (0, i+2), (1, i), (1, i+1), (1, i+2)]
                for variable in positions:
                    if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[0, i+1]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
            if(board[height-1, i+1] == MinesweeperCore.UNKNOWN_CELL):
                self.unknown_position.append((height-1, i+1))
            elif(board[height-1, i+1] > 0):
                variables = []
                positions = [(height-1, i), (height-1, i+2), (height-2, i), (height-2, i+1), (height-2, i+2)]
                for variable in positions:
                    if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[height-1, i+1]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
            if(board[i+1, 0] == MinesweeperCore.UNKNOWN_CELL):
                self.unknown_position.append((i+1, 0))
            elif(board[i+1, 0] > 0):
                variables = []
                positions = [(i, 0), (i+2, 0), (i, 1), (i+1, 1), (i+2, 1)]
                for variable in positions:
                    if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[i+1, 0]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
            if(board[i+1, height-1] == MinesweeperCore.UNKNOWN_CELL):
                self.unknown_position.append((i+1, height-1))
            elif(board[i+1, height-1] > 0):
                variables = []
                positions = [(i, height-1), (i+2, height-1), (i, height-2), (i+1, height-2), (i+2, height-2)]
                for variable in positions:
                    if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                        variables.append(variable)
                        constrained_var.add(variable)
                value = board[i+1, height-1]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
        # Constraints inside the board
        for i in range(width - 2):
            for j in range(height - 2):
                if(board[i+1, j+1] == MinesweeperCore.UNKNOWN_CELL):
                    self.unknown_position.append((i+1, j+1))
                if(board[i+1, j+1] > 0):
                    variables = []
                    positions = [(i+1, j), (i+1, j+2), (i, j), (i, j+1), (i, j+2), (i+2, j), (i+2, j+1), (i+2, j+2)]
                    for variable in positions:
                        if(board[variable] == MinesweeperCore.UNKNOWN_CELL):
                            variables.append(variable)
                            constrained_var.add(variable)
                    value = board[i+1, j+1]
                    constraint = MinesweeperConstraint(variables, value)
                    self.constraints.append(constraint)
        # Simplifies constraints
        has_subset = True
        has_simplified_version = True
        while(has_simplified_version == True):
            has_simplified_version = False
            while(has_subset == True):
                has_subset = False
                for i in range(len(self.constraints)):
                    j = i + 1
                    while(j < len(self.constraints)):
                        if(set(self.constraints[i].variables).issubset(set(self.constraints[j].variables))):
                            has_subset = True
                            variables = [var for var in self.constraints[j].variables if var not in self.constraints[i].variables]
                            value = self.constraints[j].value - self.constraints[i].value
                            self.constraints[j].variables = variables
                            self.constraints[j].value = value
                        if(set(self.constraints[j].variables).issubset(set(self.constraints[i].variables))):
                            has_subset = True
                            variables = [var for var in self.constraints[i].variables if var not in self.constraints[j].variables]
                            value = self.constraints[i].value - self.constraints[j].value
                            self.constraints[i].variables = variables
                            self.constraints[i].value = value
                        j = j + 1
                constraints = [constraint for constraint in self.constraints if constraint.variables != []]
                self.constraints = constraints
            for position in self.nobomb_position:
                for constraint in self.constraints:
                    if(position in constraint.variables):
                        has_simplified_version = True
                        constraint.variables.remove(position)
            for position in self.bomb_position:
                for constraint in self.constraints:
                    if(position in constraint.variables):
                        has_simplified_version = True
                        constraint.variables.remove(position)
                        constraint.value = constraint.value - 1
            constraints = [constraint for constraint in self.constraints if constraint.variables != []]
            self.constraints = constraints
            index_to_remove = []
            for i in range(len(self.constraints)):
                if(self.constraints[i].value == 0):
                    has_simplified_version = True
                    index_to_remove.append(i)
                    for variable in self.constraints[i].variables:
                        if(variable not in self.nobomb_position):
                            self.nobomb_position.append(variable)
                elif(self.constraints[i].value == len(self.constraints[i].variables)):
                    has_simplified_version = True
                    index_to_remove.append(i)
                    for variable in self.constraints[i].variables:
                        if(variable not in self.bomb_position):
                            self.bomb_position.append(variable)
            for i in range(len(self.constraints)):
                j = i + 1
                while(j < len(self.constraints)):
                    if(set(self.constraints[i].variables).issubset(set(self.constraints[j].variables))):
                        has_subset = True
                        has_simplified_version = True
                    if(set(self.constraints[j].variables).issubset(set(self.constraints[i].variables))):
                        has_subset = True
                        has_simplified_version = True
                    j = j + 1
            constraints = []
            for index in range(len(self.constraints)):
                if(index not in index_to_remove):
                    constraints.append(self.constraints[index])
            self.constraints = constraints
        # Removes trivials contraints from constrained_var
        non_trivials_constrained_var = []
        for variable in constrained_var:
            if(variable not in self.bomb_position and variable not in self.nobomb_position):
                non_trivials_constrained_var.append(variable)
        return self.constraints, non_trivials_constrained_var

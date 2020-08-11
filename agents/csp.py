import numpy as np
from minesweeper import MinesweeperCore
from agents.abstract_agent import AbstractAgent

class MinesweeperConstraint:
    """
    Base class for the minesweeper constraints.
    """
    def __init__(self, variables, value):
        """
        Creates a constraint i.e. (2, 1) + (2, 2) + (2, 3) = 2.

        :param variables: list of tiles with an associated constraint.
        :type variables: list of tuples (i, j).
        :param value: number of tiles in variables containing bombs.
        :type value: int.
        """
        self.variables = variables
        self.value = value

    def solvable(self, assignment):
        """
        Checks if a minesweeper constraint is still solvable after assignment.

        :param assignment: maps each tile to its value of containing a bomb (1 for true, 0 for false).
        :type assignment: dictionary.
        :return: still solvable flag
        :rtype: bool
        """
        new_value = self.value
        non_constrained = 0
        for variable in self.variables:
            if variable not in assignment:
                non_constrained = non_constrained + 1
            else:
                new_value = new_value - assignment[variable]
        return 0 <= new_value <= non_constrained


class CSPSolver:
    """
    Constraint-satisfaction problem solver for minesweeper.
    """
    def __init__(self, variables=[], constraints={}):
        """
        Initializes the constraint-satisfaction problem solver.

        :param variables: list of tiles with an associated constraint.
        :type variables: list of tuples (i, j).
        :param constraints: maps tiles to its constraints.
        :type constraints: dictionary.
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
        Adds a constraint to each of the tiles associated with that constraint.

        :param constraint: minesweeper constraint.
        :type constraint: MinesweeperConstraint.
        """
        for variable in constraint.variables:
            if variable not in self.variables:
                raise LookupError("Variable not in CSP")
            else:
                self.constraints[variable].append(constraint)

    def consistent_assignment(self, variable, assignment):
        """
        Given a new set of assignments, checks if the tile's constraints still have a solution.

        :param variable: position of a tile
        :type variable: tuple
        :param assignment: maps each tile to its value of containing a bomb (1 for true, 0 for false).
        :type assignment: dictionary.
        :return: contraints still solvable flag.
        :rtype: bool.
        """
        for constraint in self.constraints[variable]:
            if not constraint.solvable(assignment):
                return False
        return True

    def backtracking_search(self, assignment={}):
        """
        Produces solutions for the CSP.

        :param assignment: maps each tile to its value of containing a bomb (1 for true, 0 for false).
        :type assignment: dictionary.
        :return: signals the function to end search.
        :rtype: null.
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
                self.backtracking_search(local_assignment)
        return None

    def get_answers(self):
        """
        Runs backtracking search and returns solutions.

        :return: solutions to the CSP.
        :rtype: list of dictionaries (assignments).
        """
        self.backtracking_search()
        return self.solutions


class MinesweeperAgent (AbstractAgent):
    """
    Simple strategy Minesweeper agent.
    """
    def __init__(self, size, num_bombs, heuristic=True):
        """
        Initializes the Minesweeper agent considering a square board.

        :param size: size of the minesweeper board (length of one side of the board).
        :type size: int.
        :param num_bombs: number of tiles containing a bomb on the board.
        :type num_bombs: int.
        :param heuristic: whether to use the corner heuristic or not.
        :type heuristic: bool.
        """
        self.initial_position = (int(size/2), int(size/2))
        self.board_size = size
        self.num_bombs = num_bombs
        self.constraints = []
        self.nobomb_position = [self.initial_position]
        self.bomb_position = []
        self.unknown_position = []
        self.heuristic = heuristic
        self.guess_flag = False  # Verifies if the agent has made a guess

    def reset(self):
        """
        Resets agent.
        """
        self.constraints = []
        self.nobomb_position = [self.initial_position]
        self.bomb_position = []
        self.unknown_position = []
        self.guess_flag = False

    def act(self, board):
        """
        Returns a position (i, j) to play using simple strategy.

        :param board: game board.
        :type board: numpy matrix.
        """
        constraints, constrained_variables = self.read_board(board)
        # Play on first known empty tile
        self.guess_flag = False
        if len(self.nobomb_position) != 0:
            return self.nobomb_position.pop(0)
        # If all bombs are known, play on all other locations
        if len(self.bomb_position) == self.num_bombs:
            for position in self.unknown_position:
                if position not in self.bomb_position:
                    self.nobomb_position.append(position)
            return self.nobomb_position.pop(0)
        # Corner heuristic
        if self.heuristic:
            corner_position = [(0, 0), (self.board_size - 1, self.board_size - 1),
                               (0, self.board_size - 1), (self.board_size - 1, 0)]
            for position in corner_position:
                if position not in self.bomb_position and position in self.unknown_position:
                    self.guess_flag = True
                    return position
        # Gets possible answers for CSP
        csp = CSPSolver(constrained_variables, constraints)
        answers = csp.get_answers()
        # Calculates probabilities and returns the tile with highest probability of not containing a bomb
        probabilities = {}  # Variables as keys, [All cases, Cases of 0, Probability to be 0]
        for variable in constrained_variables:
            probabilities[variable] = [0, 0, 0]
        for answer in answers:
            for variable in answer:
                probabilities[variable][0] += 1
                if answer[variable] == 0:
                    probabilities[variable][1] += 1
        best_position = [(0, 0), 0]
        for variable in constrained_variables:
            if probabilities[variable][0] != 0:
                probabilities[variable][2] = probabilities[variable][1] / probabilities[variable][0]
            if probabilities[variable][2] > best_position[1] and variable not in self.bomb_position:
                best_position[1] = probabilities[variable][2]
                best_position[0] = variable
        if best_position[1] == 0:
            playable_positions = [position for position in self.unknown_position if position not in self.bomb_position]
            rand_index = np.random.randint(len(playable_positions))
            best_position[0] = playable_positions[rand_index]
        self.guess_flag = True
        return best_position[0]

    def read_board(self, board):
        """
        Reads minesweeper board. Returns simplified constraints and all variables constrained.

        :param board: game board.
        :type board: numpy matrix.
        :return: (simplified constraints, variables associated with constraints)
        :rtype: (list of MinesweeperConstraints, list of tuples)
        """
        self.constraints = []
        self.unknown_position = []
        constrained_var = set()
        height = board.shape[0]
        width = board.shape[1]
        # Creates constraints
        tiles = [(i, j) for i in range(height) for j in range(width)]
        for tile in tiles:
            if board[tile] == MinesweeperCore.UNKNOWN_CELL:
                self.unknown_position.append(tile)
            elif board[tile] > 0:
                variables = []
                neighbors = []
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        neighbor = (tile[0] + i, tile[1] + j)
                        if neighbor != tile and 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                            neighbors.append(neighbor)
                for neighbor in neighbors:
                    if board[neighbor] == MinesweeperCore.UNKNOWN_CELL:
                        variables.append(neighbor)
                        constrained_var.add(neighbor)
                value = board[tile]
                constraint = MinesweeperConstraint(variables, value)
                self.constraints.append(constraint)
        # Simplifies constraints
        has_subset = True
        has_simplified_version = True
        while has_simplified_version:
            has_simplified_version = False
            # Removes subsets of constraints
            while has_subset:
                has_subset = False
                for i in range(len(self.constraints)):
                    for j in range(len(self.constraints)):
                        if i != j and set(self.constraints[i].variables).issubset(set(self.constraints[j].variables)):
                            has_subset = True
                            variables = [var for var in self.constraints[j].variables
                                         if var not in self.constraints[i].variables]
                            value = self.constraints[j].value - self.constraints[i].value
                            self.constraints[j].variables = variables
                            self.constraints[j].value = value
                constraints = [constraint for constraint in self.constraints if constraint.variables != []]
                self.constraints = constraints
            # Removes known empty tiles
            for position in self.nobomb_position:
                for constraint in self.constraints:
                    if position in constraint.variables:
                        has_simplified_version = True
                        constraint.variables.remove(position)
            # Removes known tiles with bombs
            for position in self.bomb_position:
                for constraint in self.constraints:
                    if position in constraint.variables:
                        has_simplified_version = True
                        constraint.variables.remove(position)
                        constraint.value = constraint.value - 1
            constraints = [constraint for constraint in self.constraints if constraint.variables != []]
            self.constraints = constraints
            # Removes tiles in constraints with value 0 or equal to number of tiles in constraint
            for constraint in self.constraints:
                if constraint.value == 0:
                    has_simplified_version = True
                    for variable in constraint.variables:
                        self.nobomb_position.append(variable)
                    constraints.remove(constraint)
                elif constraint.value == len(constraint.variables):
                    has_simplified_version = True
                    for variable in constraint.variables:
                        self.bomb_position.append(variable)
                    constraints.remove(constraint)
            self.constraints = constraints
            for i in range(len(self.constraints)):
                for j in range(len(self.constraints)):
                    if i != j and set(self.constraints[i].variables).issubset(set(self.constraints[j].variables)):
                        has_simplified_version = True
                        has_subset = True
        # Removes trivial contraints from constrained_var
        non_trivials_constrained_var = []
        for variable in constrained_var:
            if variable not in self.bomb_position and variable not in self.nobomb_position:
                non_trivials_constrained_var.append(variable)
        return self.constraints, non_trivials_constrained_var

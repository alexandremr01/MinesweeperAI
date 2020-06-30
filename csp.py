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

class MinesweeperCSP:
    """
    Constraint-satisfaction problem for minesweeper.
    """
    def __init__(self, variables, constraints):
        """
        :type variables: list of tuples (i, j).
        :type domains: list.
        :type constraints: dictionary.
        """
        self.variables = variables
        self.domains = [0, 1]
        self.constraints = constraints
        for variable in self.variables:
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

    def backtracking_search(self, assignment):
        """
        Gives a solution for a CSP

        :param assignment: Maps a set of variables to a value
        :type assignment: dictionary
        """
        if len(assignment) == len(self.variables):
            return assignment
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
                if result is not None:
                    return result
        return None

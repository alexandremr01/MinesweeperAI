import random
import numpy as np


class MinesweeperCore:
    """
    Represents the core game rules.
    """
    BOMB = -2
    UNKNOWN_CELL = -1
    CLEAR_CELL = 0

    def __init__(self, height, width, num_bombs, win_threshold=1.0):
        """
        Defines game parameters and initializes a new minesweeper board.
        Where the x coordinate represents the rows ranging from 0 to self.height - 1
        and y coordinate represents the columns ranging from 0 to self.width - 1.

        :param height: height of the board.
        :type height: int.
        :param width: width of the board.
        :type width: int.
        :param num_bombs: number of bomb on the board.
        :type num_bombs: int.
        :param win_threshold: percentage of the game board discovered to consider victory.
        :type win_threshold: float.
        """
        self.height = height
        self.width = width
        self.num_bombs = num_bombs
        self.win_threshold = win_threshold
        self.table = np.matrix(np.full((self.height, self.width), self.UNKNOWN_CELL))
        self.victory = False
        self.still_playing = True
        self.bomb_positions = []
        self.unexplored = self.height * self.width - self.num_bombs
        self.first_play = True
        for i in range(self.num_bombs):
            while True:
                x = random.randint(0, self.height - 1)
                y = random.randint(0, self.width - 1)
                if (x, y) not in self.bomb_positions:
                    self.bomb_positions.append((x, y))
                    break

    def reset(self):
        """
        Resets the minesweeper board.

        :return: reset board.
        :rtype: numpy matrix.
        """
        self.table = np.matrix(np.full((self.height, self.width), self.UNKNOWN_CELL))
        self.victory = False
        self.still_playing = True
        self.bomb_positions = []
        self.unexplored = self.height * self.width - self.num_bombs
        self.first_play = True
        for i in range(self.num_bombs):
            while True:
                x = random.randint(0, self.height - 1)
                y = random.randint(0, self.width - 1)
                if (x, y) not in self.bomb_positions:
                    self.bomb_positions.append((x, y))
                    break
        return self.table

    def isVictory(self):
        """
        Returns the state of the game.

        :return: victory flag.
        :rtype: bool.
        """
        return self.victory

    def isPlaying(self):
        """
        Returns if the agent is still playing the game.

        :return: still playing flag.
        :rtype: bool.
        """
        return self.still_playing

    def play(self, x, y):
        """
        Executes discovering action on the (x, y) position on the game board and updates it accordingly.

        :param x: x coordinate on the game board.
        :type x: int.
        :param y: y coordinate on the game board.
        :type y: int.
        :return: output of the discovering decision.
            True for successful action and False for incorrect action.
        :rtype: bool.
        """
        if not (0 <= x < self.height and 0 <= y < self.width):
            return False
        if (self.table[x, y] != self.UNKNOWN_CELL) or (not self.isPlaying()):
            return False
        if (x, y) in self.bomb_positions:
            if self.first_play is True:
                self.reset()
                return self.play(x, y)
            self.still_playing = False
            return True
        self.first_play = False
        neighbour_bombs = self.neighbour_bombs(x, y)
        self.unexplored -= 1
        self.table[x, y] = neighbour_bombs
        if neighbour_bombs == 0:
            displacement = [-1, 0, 1]
            for i in displacement:
                for j in displacement:
                    self.play(x + i, y + j)
        self.verify_win()
        return True

    def verify_win(self):
        """
        Verifies the current state of the game and updates the state variables accordingly.
        """
        if self.unexplored > (1 - self.win_threshold) * self.height * self.width:
            return False
        self.victory = True
        self.still_playing = False

    def neighbour_bombs(self, x, y):
        """
        Counts how many neighbours of (x,y) are bombs.

        :param x: x coordinate on the game board.
        :type x: int.
        :param y: y coordinate on the game board.
        :type y: int.
        :return: number of bombs adjacent to the position (x,y).
        :rtype: int.
        """
        neighbours = 0
        displacement = [-1, 0, 1]
        for i in displacement:
            for j in displacement:
                if (x + i, y + j) in self.bomb_positions:
                    neighbours += 1
        return neighbours

    def get_board(self, xray=False):
        """
        Returns the board.

        :param xray: if the board should show all bomb positions or not.
        :type xray: bool.
        :return: game board.
        :rtype: numpy matrix.
        """
        if not xray:
            return self.table
        else:
            xray_table = np.copy(self.table)
            for position in self.bomb_positions:
                xray_table[position[0], position[1]] = self.BOMB
            return xray_table

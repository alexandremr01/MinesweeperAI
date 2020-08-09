import numpy as np
from minesweeper import MinesweeperCore


class MinesweeperEnvironment:
    """
    Reinforcement Learning wrapper to the core game.
    """
    def __init__(self, height, width, num_bombs, win_threshold=1.0):
        """
        Initializes the RL wrapper.

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
        self.game = MinesweeperCore(height, width, num_bombs, win_threshold)

    def reset(self):
        """
        Resets the game board.

        :return: reset board.
        :rtype: numpy matrix.
        """
        return self.game.reset()

    def reward_engineering(self):
        """
        Computes the reward and returns it.

        :return: reward of RL.
        :rtype: int.
        """
        reward = 0
        if not self.is_finished():
            reward = 1
        elif self.is_victory():
            open_table = self.game.table != self.game.UNKNOWN_CELL
            reward = np.sum(open_table)
        return reward

    def step(self, x, y):
        """
        Performs discovering action on (x, y) tile and computes the reward produced.

        :param x: x coordinate on the game board.
        :type x: int.
        :param y: y coordinate on the game board.
        :type y: int.
        :return: (game board after the action was executed,
            reward produced by the action,
            flag corresponding to the condition still playing)
        :rtype: (numpy matrix, int, bool).
        """
        self.game.play(x, y)
        next_state = self.game.get_board()
        reward = self.reward_engineering()
        done = not self.game.still_playing
        return next_state, reward, done

    def get_state(self, xray=False):
        """
        Returns the board.

        :param xray: if the board should show all bomb positions or not.
        :type xray: bool.
        :return: game board.
        :rtype: numpy matrix.
        """
        return self.game.get_board(xray)

    def is_finished(self):
        """
        Returns if the agent is still playing the game.

        :return: is finished flag.
        :rtype: bool.
        """
        return not self.game.still_playing

    def is_victory(self):
        """
        Returns the state of the game.

        :return: victory flag.
        :rtype: bool.
        """
        return self.game.isVictory()

    def get_open_percentage(self):
        """
        Computes the percentage of the board that has been uncovered.

        :return: uncovered percentage of the board.
        :rtype: float.
        """
        open_table = self.game.table != self.game.UNKNOWN_CELL
        return np.sum(open_table) / (self.height * self.width - self.num_bombs)

    def print_board(self, xray=False):
        """
        Prints the current state of the board.

        :param xray: if the board should show all bomb positions or not.
        :type xray: bool.
        """
        print_table = self.game.get_board(xray)
        for i in range(self.height):
            for j in range(self.width):
                if print_table[i, j] == self.game.BOMB:
                    print("#", end=' ')
                elif print_table[i, j] == self.game.UNKNOWN_CELL:
                    print(".", end=' ')
                elif print_table[i, j] == self.game.CLEAR_CELL:
                    print("0", end=' ')
                else:
                    print(int(print_table[i, j]), end=' ')
            print("")

    def print_bomb_positions(self):
        """
        Prints all known bomb positions.
        """
        print("Bomb positions: " + str(self.game.bomb_positions))

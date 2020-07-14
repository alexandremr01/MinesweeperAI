import numpy as np
import random
from minesweeper import MinesweeperCore
class MinesweeperEnvironment:
    """
    Reinforcement Learning wrapper to the core game.
    """
    def __init__(self, height, width, num_bombs, win_threshold = 0.8):
      self.height = height
      self.width = width
      self.num_bombs = num_bombs
      self.game = MinesweeperCore(height, width, num_bombs, win_threshold)

    def reset(self):
      return self.game.reset()

    def reward_engineering(self):
      reward = 0
      if not self.is_finished():
        reward = 1
      elif self.is_victory():
        open_table = self.game.table != self.game.UNKNOWN_CELL
        reward = np.sum(open_table)
      return reward

    def step(self, x, y):
      self.game.play(x, y)
      next_state = self.game.get_board()
      reward = self.reward_engineering()
      done = not self.game.still_playing
      return next_state, reward, done

    def get_state(self, xray=False):
      return self.game.get_board(xray)

    def is_finished(self):
      return not self.game.still_playing

    def is_victory(self):
      return self.game.isVictory()

    def get_open_percentage(self):
      open_table = self.game.table != self.game.UNKNOWN_CELL
      return np.sum(open_table) / (self.height * self.width)

    def print_board(self, xray = False):
      print_table = self.game.get_board(xray)
      for i in range(self.height):
        for j in range(self.width):
          if(print_table[i, j] == self.game.BOMB):
            print("#",end=' ')
          elif (print_table[i, j] == self.game.UNKNOWN_CELL):
            print(".",end=' ')
          elif (print_table[i, j] == self.game.CLEAR_CELL):
            print("0",end=' ')
          else:
            print(int(print_table[i, j]), end=' ')
        print("")
      print("Bomb positions: " + str(self.game.bomb_positions))

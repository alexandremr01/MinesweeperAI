import numpy as np
import random

class MinesweeperCore:
    """
    Represents the core game rules.
    """
    BOMB = -2
    UNKNOWN_CELL = -1
    CLEAR_CELL = 0

    def __init__(self, height, width, num_bombs, win_threshold=1.0):
      """
      Basic game configuration and call board initializer.
      """
      self.height = height
      self.width = width
      self.num_bombs = num_bombs
      self.win_threshold = win_threshold
      self.reset()

    def reset(self):
      """
      Resets board.
      """
      self.table = np.matrix(np.full((self.height, self.width), self.UNKNOWN_CELL))
      self.victory = False
      self.still_playing = True
      self.bomb_positions = []
      self.unexplored = self.height*self.width - self.num_bombs
      self.first_play = True
      for i in range(self.num_bombs):
        while True:
          x = random.randint(0, self.height-1)
          y = random.randint(0, self.width-1)
          if (x, y) not in self.bomb_positions:
            self.bomb_positions.append((x, y))
            break
      return self.table

    def isVictory(self):
      return self.victory

    def isPlaying(self):
      return self.still_playing

    def play(self, x, y):
      if (x>= self.height or y>=self.width or x<0 or y<0):
        return False
      if (self.table[x, y] != self.UNKNOWN_CELL) or (self.still_playing == False):
        return False
      if (x, y) in self.bomb_positions:
        if self.first_play is True:
          self.reset()
          return self.play(x, y)
        self.still_playing = False
        return True
      self.first_play = False
      neighbour_bombs = self.neighbour_bombs(x, y)
      self.unexplored-=1
      self.table[x, y] = neighbour_bombs
      if neighbour_bombs == 0:
        displacement = [-1, 0, 1]
        for i in displacement:
          for j in displacement:
            self.play(x+i, y+j)
      self.verify_win()
      return True

    def verify_win(self):
      if self.unexplored > (1-self.win_threshold)*self.height*self.width:
        return False
      self.victory = True
      self.still_playing = False

    def neighbour_bombs(self, x, y):
      """
      Counts how many neighbour of (x,y) are bombs.
      """
      neighbours = 0
      displacement = [-1, 0, 1]
      for i in displacement:
        for j in displacement:
          if (x+i, y+j) in self.bomb_positions:
            neighbours += 1
      return neighbours

    def get_board(self, xray = False):
      """
      Returns the board.
      """
      if xray == False:
        return self.table
      else:
        xray_table = np.copy(self.table)
        for position in self.bomb_positions:
          xray_table[position[0], position[1]] = self.BOMB
        return xray_table
        

import numpy as np
import random
from minesweeper_environment import MinesweeperEnvironment
from minesweeper import MinesweeperCore
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os

# This script analyzes log (old log format)

file = open('log', 'r')
content = file.read()
lista = content.split('],[')
j = 0
for i in range(1, len(lista)-1):
  jogo = lista[i].split('), (')
  jogo[0] = jogo[0][1:]
  jogo[len(jogo)-1] = jogo[len(jogo)-1][0:-1]
  if len(jogo) != len(set(jogo)):
    j+=1
print(j)
print(len(lista))
import os
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import pylab
from minesweeper import MinesweeperCore
from minesweeper_environment import MinesweeperEnvironment

NUM_EPISODES = 500  # Number of episodes used for training
fig_format = 'png'  # Format used for saving matplotlib's figures

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initiating the environment
side = 9
bombs = 10
minesweeper = MinesweeperEnvironment(side, side, bombs)

# Creating the DQN agent
agent = DQNAgent(side)

# Checking if weights from previous learning session exists
if os.path.exists('minesweeper.h5'):
    print('Loading weights from previous learning session.')
    agent.load("minesweeper.h5")
else:
    print('No weights found from previous learning session.')
done = False
batch_size = 16  # batch size used for the experience replay
return_history = []

out_file = open('log', 'w')

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = minesweeper.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    moves = []
    for plays in range(1, 500):
        # Select action
        x, y = agent.act(state)
        action = (x, y)
        moves.append(action)
        next_state, reward, done = minesweeper.step(x, y)
        #minesweeper.print_board()
        #print(reward)
        #input("")
        # Appending this experience to the experience replay buffer
        agent.append_experience(state, action, reward, next_state, done)
        state = next_state
        #minesweeper.print_board()
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            out_file.write(str(minesweeper.game.bomb_positions)+'\n')
            out_file.write(str(moves)+'\n')
            print("episode: {}/{}, plays: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, plays, cumulative_reward, agent.epsilon))
            break
        # We only update the policy if we already have enough experience in memory
        if len(agent.replay_buffer) > 2 * batch_size:
            loss = agent.replay(batch_size)
    return_history.append(cumulative_reward)
    agent.update_epsilon()
    # Every 10 episodes, update the plot for training monitoring
    if episodes % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('dqn_training.' + fig_format, fig_format=fig_format)
        # Saving the model to disk
        agent.save("minesweeper.h5")
plt.pause(1.0)

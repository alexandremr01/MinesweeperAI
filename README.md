# MinesweeperAI

- [Directory structure](#directory_structure)
- [Dataset generation](#dataset_generation)
- [Neural network training](#neural_network_training)
- [Agent visualizer](#agent_visualizer)
- [Evaluator](#evaluator)

- [Team](#team)
- [License](#license)


> **_NOTE:_** Bold words are variables from .py files.

# Directory structure
    .
    ├── agents                  # 3 agents: CSP, Imitator and Enhanced
    ├── dataset                 # Dataset is generated and stored here
    └── README.md

# Dataset generation

The datagen script automatically generate and store dataset in batches. Dataset size is defined as number of board-action pairs. Set in the datagen script the total desired size and size per file, an then run it.

After generating the desired amount (recommended: at least 100k), use the preprocess_dataset script to remove non-unique boards and unite all unique in a single file.

Usage:

        python datagen.py
        python preprocess_dataset.py

# Agent visualizer

For a given **size** x **size** board with **bombs** number of mines, runs one match and prints the board configuration for each move. Assign the following names to **agent_name** to choose an agent:

- h_csp for heuristic CSP.
- nh_csp for non-heuristic CSP.
- l4ms for supervised learning agent.

The CSP can play in any board size. The remaining agents can only play in 8x8 board.

Usage:

        python play_one_match.py

# Evaluator

The game is set similar to Agent visualizer.

The single evaluator script runs **NUM_EPISODES** games and gives three metrics: number of plays until defeat, board open percentage, and win rate.

The multiple evaluator script gives the same metrics but for three same dimension boards with different number of mines.

Usage:

        python evaluator.py
        python multi_evaluator.py

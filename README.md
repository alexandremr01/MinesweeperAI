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
    ├── agents                  # 2 agents: CSP and L4MS (4-Layer MineSweeper, standing for the 4 convolutional layers)
    ├── dataset                 # Dataset is generated and stored here
    ├── logs                    # Tensorboard logs
    ├── results                 # Trained weights and some images
    └── README.md

# Dataset generation

The datagen script automatically generates and stores the dataset in batches. Dataset size is defined as the number of board-action pairs. Set in the datagen script the desired total size and size per file then run it.

After generating the desired amount (recommended: at least 100k, 8.45 million were used for the research), use the preprocess_dataset script to remove non-unique boards and unite all uniques in a single file.

Usage:

        python datagen.py
        python preprocess_dataset.py

# Neural network training

The train_supervised.py script 

Usage:

        python train_supervised.py

# Agent visualizer

For a given **size** x **size** board with **bombs** number of mines, runs one match and prints the board configuration for each move. Assign the following names to **agent_name** to choose an agent:

- h_csp for heuristic CSP.
- nh_csp for non-heuristic CSP.
- l4ms for supervised learning agent.

The CSP can play in any board size. The remaining agents can only play in an 8x8 board.

Usage:

        python play_one_match.py

# Evaluator

The game is set similar to Agent visualizer.

The single evaluator script runs **NUM_EPISODES** games and returns three metrics: number of plays until defeat, board open percentage, and win rate.

The multiple evaluator script returns the same metrics but for three same dimension boards with different numbers of mines.

Usage:

        python single_evaluator.py
        python multi_evaluator.py

# MinesweeperAI

- [Directory structure](#directory_structure)
- [Dataset generation](#dataset_generation)
- [Neural network training](#neural_network_training)
- [Evaluator](#evaluator)

- [Team](#team)
- [License](#license)



# Directory structure
    .
    ├── agents                  # 3 agents: CSP, Imitator and Enhanced
    ├── dataset                 # Dataset is generated and stored here
    └── README.md

# Dataset generation

The datagen script automatticaly generate and store dataset in batches. Dataset size is defined as number of board-action pairs. Set in the datagen script the total desired size and size per file, an then run it. 

After generating the desired amount (recommended: at least 100k), use the preprocess_dataset script to remove non-unique boards and unite all unique in a single file. 

Usage:

        python datagen.py
        python preprocess_dataset.py

# Neural network training

The train_supervised.py script 

Usage:

        python datagen.py
        python preprocess_dataset.py






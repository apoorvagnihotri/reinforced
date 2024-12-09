#!/bin/bash

# Define the list of epsilon values
SEED_VALUES=(10 22 50 44 5)

# Iterate over each epsilon value and run the script
for SEED in "${SEED_VALUES[@]}"
do
    echo "Running PPO.py with seed=$SEED"
    python PPO.py --seed $SEED --eps 0.2
done

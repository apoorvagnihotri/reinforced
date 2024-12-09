#!/bin/bash

# Define the list of epsilon values
EPS_VALUES=(0.01 0.1 0.2 0.5 0.75)

# Iterate over each epsilon value and run the script
for EPS in "${EPS_VALUES[@]}"
do
    echo "Running PPO.py with eps=$EPS"
    python PPO.py --eps $EPS
done

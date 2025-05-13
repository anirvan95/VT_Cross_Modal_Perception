#!/bin/bash

# Path to the virtual environment
VENV_PATH="/home/robotac_ws0/PhD_VisioHapticExploration/Projects/Visuo_Tactile_Cross_Modal_Perception/.venv/"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# List of Python scripts
SCRIPTS=(
    "evaluate_evolution_aligned_v2t.py"
    "evaluate_evolution_aligned_t2v.py"
    "evaluate_evolution_aligned_wocm.py"
    "evaluate_evolution_aligned_joint.py"
)

# Base path to the scripts
BASE_PATH="/home/robotac_ws0/PhD_VisioHapticExploration/Projects/Visuo_Tactile_Cross_Modal_Perception"

# Loop through and execute each script
for script in "${SCRIPTS[@]}"; do
    echo "Running $script"
    python "$BASE_PATH/$script"
done

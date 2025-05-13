#!/bin/bash

# Path to the virtual environment
VENV_PATH="/media/robotac22/DataVault/Cross Modal Repo/.venv/"

# Python program name
PYTHON_PROGRAM="/media/robotac22/DataVault/Cross Modal Repo/train_cross_modal.py"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Parameter combinations for crossmodal_weight and crossmodal_rate
params=(
  "1 0.25"
  "2.5 0.25"
  "5 0.25"
  "1 0.5"
  "2.5 0.5"
  "5 0.5"
  "1 0.75"
  "2.5 0.75"
  "5 0.75"
)

# Iterate over parameter sets
for i in "${!params[@]}"; do
  # Parse individual parameters
  set -- ${params[$i]}
  crossmodal_weight=$1
  crossmodal_rate=$2

  # Build and run the command
  echo "Running experiment $((i+1)) with parameters:"
  echo "  crossmodal_weight=$crossmodal_weight, crossmodal_rate=$crossmodal_rate"

  python $PYTHON_PROGRAM \
    --crossmodal_weight $crossmodal_weight \
    --crossmodal_rate $crossmodal_rate

  # Check exit status
  if [ $? -ne 0 ]; then
    echo "Experiment $((i+1)) failed. Exiting."
    deactivate
    exit 1
  fi
done

# Deactivate the virtual environment
deactivate

echo "All experiments completed successfully."

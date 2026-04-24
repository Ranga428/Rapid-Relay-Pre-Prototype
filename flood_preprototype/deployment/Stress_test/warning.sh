#!/bin/bash
# Move to the directory where the script is located
cd "$(dirname "$0")"

# Activate the virtual environment (two levels up)
source ../../floodenv/bin/activate

# Run the simulation for WARNING tier
python3 simulation.py --force-alerts --tiers warning

# Keep terminal open
read -p "Press enter to continue..."
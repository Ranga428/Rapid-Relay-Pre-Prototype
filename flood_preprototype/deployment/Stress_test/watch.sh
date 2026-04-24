#!/bin/bash
# Move to the directory where the script is located
cd "$(dirname "$0")"

# Activate the virtual environment (two levels up)
source ../../floodenv/bin/activate

# Run the simulation for WATCH tier
python3 simulation.py --force-alerts --tiers watch

# Keep terminal open (optional, depends on terminal settings)
read -p "Press enter to continue..."
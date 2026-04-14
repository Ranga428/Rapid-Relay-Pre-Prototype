#!/bin/bash
# Always start in the folder where this script is located (deployment/)
cd "$(dirname "$0")"

# Activate the virtual environment (two levels up from deployment/)
source ../../floodenv/Scripts/activate

# Move into the scripts folder where Rapid_Relay_Showcase.py lives
cd "$(dirname "$0")/../scripts"

# Run the realtime pipeline
python Rapid_Relay_Showcase.py --realtime --predict-on-insert
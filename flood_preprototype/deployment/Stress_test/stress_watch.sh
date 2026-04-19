#!/bin/bash
# Always start in the folder where this script is located (deployment/Stress_test/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment (three levels up from Stress_test/)
source "$SCRIPT_DIR/../../../floodenv/bin/activate"

# Move into the scripts folder where stress_test_gen.py lives
cd "$SCRIPT_DIR/../../scripts"

# Default count = 1. Override by calling: ./stress_watch.sh 31
COUNT=${1:-1}

echo ""
echo "============================================================"
echo " Rapid Relay EWS - Stress Test Supabase Insert [WATCH]"
echo " Count: $COUNT"
echo "============================================================"
echo ""

python stress_test_gen.py --scenario watch --supabase --count "$COUNT" || { echo "ERROR: WATCH insert failed."; exit 1; }

echo ""
echo " DONE - $COUNT WATCH row(s) inserted into Supabase."
echo ""

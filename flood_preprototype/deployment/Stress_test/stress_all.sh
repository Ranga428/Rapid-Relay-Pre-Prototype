#!/bin/bash
# Always start in the folder where this script is located (deployment/Stress_test/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment (three levels up from Stress_test/)
source "$SCRIPT_DIR/../../../floodenv/bin/activate"

# Move into the scripts folder where stress_test_gen.py lives
cd "$SCRIPT_DIR/../../scripts"

# Default count = 1. Override by calling: ./stress_all.sh 31
# Note: --scenario all inserts 4 rows per count (one per tier)
#       so ./stress_all.sh 31 = 124 total rows
COUNT=${1:-1}

echo ""
echo "============================================================"
echo " Rapid Relay EWS - Stress Test Supabase Insert [ALL]"
echo " Count: $COUNT per tier  (x4 tiers = $((COUNT * 4)) total rows)"
echo "============================================================"
echo ""

python stress_test_gen.py --scenario all --supabase --count "$COUNT" || { echo "ERROR: ALL insert failed."; exit 1; }

echo ""
echo " DONE - $COUNT row(s) per tier inserted into Supabase."
echo ""

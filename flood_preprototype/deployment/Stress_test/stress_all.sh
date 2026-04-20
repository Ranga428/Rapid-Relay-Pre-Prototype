#!/bin/bash
# Always start in the folder where this script is located (deployment/Stress_test/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment
source "$SCRIPT_DIR/../../../floodenv/bin/activate"

# ── Resolve key paths ────────────────────────────────────────────────────
SCRIPTS_DIR="$SCRIPT_DIR/../../scripts"
SHOWCASE_DIR="$SCRIPT_DIR/../../showcase"
STRESS_DIR="$SCRIPT_DIR/../../data/stress_test"

echo ""
echo "============================================================"
echo " Rapid Relay EWS - Stress Test Pipeline [ALL TIERS]"
echo " Runs full 3-step pipeline for each tier in sequence:"
echo "   CLEAR -> WATCH -> WARNING -> DANGER"
echo " Step 1 : Generate 30-day stress CSV per tier"
echo " Step 2 : Seed showcase_sensor.csv with context (no alert)"
echo " Step 3 : Insert 1 live Supabase row (triggers alert)"
echo "============================================================"
echo ""

run_tier() {
    local TIER="$1"
    local NUM="$2"

    echo "── ${TIER} (${NUM}/4) ─────────────────────────────────────────────"

    echo "[1/3] Generating 30-day ${TIER} stress CSV..."
    cd "$SCRIPTS_DIR"
    python stress_test_gen.py --scenario "$TIER" --days 30 --csv || { echo "ERROR: ${TIER} CSV generation failed."; exit 1; }
    echo "      Done."

    echo "[2/3] Seeding showcase_sensor.csv with ${TIER} context (no alert)..."
    cd "$SHOWCASE_DIR"
    python showcase_predict.py --seed "${STRESS_DIR}/stress_${TIER}.csv" || { echo "ERROR: ${TIER} seed failed."; exit 1; }
    echo "      Done."

    echo "[3/3] Inserting 1 live ${TIER} row into Supabase (date = day 31 of context)..."
    cd "$SCRIPTS_DIR"
    python stress_test_gen.py --scenario "$TIER" --supabase --count 1 --after-csv "${STRESS_DIR}/stress_${TIER}.csv" || { echo "ERROR: ${TIER} Supabase insert failed."; exit 1; }
    echo "      Done."
    echo ""
}

run_tier "clear"   1
run_tier "watch"   2
run_tier "warning" 3
run_tier "danger"  4

echo "============================================================"
echo " DONE - ALL TIERS stress test pipeline complete."
echo " Each tier: 30-row context seeded + 1 live Supabase row inserted"
echo " Live rows : 4 inserted into Supabase (alerts triggered per tier)"
echo "============================================================"
echo ""

#!/bin/bash
# Always start in the folder where this script is located (deployment/Stress_test/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment
source "$SCRIPT_DIR/../../../floodenv/bin/activate"

# ── Resolve key paths ────────────────────────────────────────────────────
SCRIPTS_DIR="$SCRIPT_DIR/../../scripts"
SHOWCASE_DIR="$SCRIPT_DIR/../../showcase"
STRESS_CSV="$SCRIPT_DIR/../../data/stress_test/stress_warning.csv"

echo ""
echo "============================================================"
echo " Rapid Relay EWS - Stress Test Pipeline [WARNING]"
echo " Step 1 : Generate 30-day stress CSV"
echo " Step 2 : Seed showcase_sensor.csv with context (no alert)"
echo " Step 3 : Insert 1 live Supabase row (triggers alert)"
echo "============================================================"
echo ""

# ── STEP 1: Generate 30-day stress CSV ──────────────────────────────────
echo "[1/3] Generating 30-day WARNING stress CSV..."
cd "$SCRIPTS_DIR"
python stress_test_gen.py --scenario warning --days 30 --csv || { echo "ERROR: CSV generation failed."; exit 1; }
echo "      Done. CSV saved to data/stress_test/stress_warning.csv"
echo ""

# ── STEP 2: Seed showcase_sensor.csv with 30 rows of context ─────────────
echo "[2/3] Seeding showcase_sensor.csv with 30-day context..."
echo "      NOTE: Writes directly to showcase_sensor.csv — no Supabase,"
echo "            no prediction, no alert triggered."
cd "$SHOWCASE_DIR"
python showcase_predict.py --seed "$STRESS_CSV" || { echo "ERROR: Seed step failed."; exit 1; }
echo "      Done. Rolling-window context is now available."
echo ""

# ── STEP 3: Insert 1 live row into Supabase (triggers the real alert) ────
echo "[3/3] Inserting 1 live WARNING row into Supabase (date = day 31 of context)..."
echo "      NOTE: This row lands on top of the 30-day context and will"
echo "            produce an accurate tier prediction + trigger the alert."
cd "$SCRIPTS_DIR"
python stress_test_gen.py --scenario warning --supabase --count 1 --after-csv "$STRESS_CSV" || { echo "ERROR: Supabase insert failed."; exit 1; }
echo "      Done. Live WARNING row inserted into Supabase."
echo ""

echo "============================================================"
echo " DONE - WARNING stress test pipeline complete."
echo " Context seed  : 30 rows written to showcase_sensor.csv"
echo " Live row      : Inserted into Supabase (alert triggered)"
echo "============================================================"
echo ""

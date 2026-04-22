#!/bin/bash

# ============================================================
# cleanup_latest.sh
# Location: /home/kali/Rapid-Relay-Pre-Prototype/flood_preprototype/deployment/
# Purpose:  Reset state by deleting alert JSON files and
#           removing the latest row(s) from sensor/prediction CSVs
# ============================================================

set -euo pipefail

BASE="/home/kali/Rapid-Relay-Pre-Prototype/flood_preprototype"

# ── Helpers ─────────────────────────────────────────────────

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*"; }
err()  { echo "[ERROR] $*" >&2; }

delete_file() {
    local file="$1"
    if [ -f "$file" ]; then
        rm -f "$file"
        log "Deleted: $file"
    else
        warn "Not found (skipped): $file"
    fi
}

# Remove the last non-empty row of a CSV while keeping the header intact.
# A backup (.bak) is created before any modification.
delete_last_csv_row() {
    local file="$1"

    if [ ! -f "$file" ]; then
        warn "Not found (skipped): $file"
        return
    fi

    local total_lines
    total_lines=$(wc -l < "$file")

    # Need at least a header + 1 data row
    if [ "$total_lines" -lt 2 ]; then
        warn "Nothing to remove (file has $total_lines line(s)): $file"
        return
    fi

    # Back up before modifying
    cp "$file" "${file}.bak"
    log "Backup created: ${file}.bak"

    # Drop the last line (handles files with or without a trailing newline)
    local keep=$(( total_lines - 1 ))
    head -n "$keep" "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"

    log "Removed last row from: $file  (was $total_lines lines, now $keep)"
}

# ── 1. Delete alert channel state files ─────────────────────

log "--- Cleaning alert channel state ---"
delete_file "$BASE/alerts/Channels/last_posted.json"
delete_file "$BASE/alerts/Channels/last_telegram_sent.json"

# ── 2. Trim latest row from sensor CSVs ─────────────────────

log "--- Trimming latest row from sensor CSVs ---"
delete_last_csv_row "$BASE/data/sensor/combined_sensor_context.csv"
delete_last_csv_row "$BASE/data/sensor/obando_environmental_data.csv"
delete_last_csv_row "$BASE/data/sensor/obando_sensor_data.csv"

# ── 3. Trim latest row from prediction CSV ──────────────────

log "--- Trimming latest row from prediction CSV ---"
delete_last_csv_row "$BASE/predictions/flood_xgb_sensor_predictions.csv"

log "--- Cleanup complete ---"
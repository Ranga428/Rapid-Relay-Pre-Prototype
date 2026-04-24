"""
simulation.py
===============
Rapid Relay EWS — Full Simulation Pipeline Orchestrator

Per-tier sequence:
  1. stress_test_gen.py  --scenario <tier> --csv --days <N>
  2. swmm_viz_pyswmm.py  <inp>
  3. XGB_Predict.py      --data <stress_csv>
  4. Alert.py            --csv  <predict_csv>

Usage:
  python simulation.py                         # all 4 tiers, 30 days
  python simulation.py --tiers clear watch     # subset
  python simulation.py --days 7                # shorter sim
  python simulation.py --skip-viz              # skip animation (slow)
  python simulation.py --dry-run               # print commands, no exec
  python simulation.py --stride 6              # viz frame stride
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# Script lives at: flood_preprototype/deployment/simulation.py
# Project root  at: flood_preprototype/
ROOT        = Path(__file__).resolve().parent.parent   # flood_preprototype/
SCRIPTS_DIR = ROOT / "scripts"
SWMM_DIR    = ROOT / "data" / "swmm"
ALERTS_DIR  = ROOT / "alerts"
STRESS_DIR  = ROOT / "data" / "stress_test"
PREDICT_DIR = ROOT / "predictions"
INP_FILE    = SWMM_DIR / "obando_bulacan.inp"

STRESS_GEN  = SCRIPTS_DIR / "stress_test_gen.py"
VIZ_SCRIPT  = SWMM_DIR   / "swmm_viz_pyswmm.py"
PREDICT     = SCRIPTS_DIR / "XGB_Predict.py"
ALERT       = ALERTS_DIR  / "Alert.py"

ALL_TIERS   = ["clear", "watch", "warning", "danger"]

PYTHON      = sys.executable   # same venv that launched this script

# ── Helpers ────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {msg}\n{line}")


def run(cmd: list[str], label: str, dry_run: bool) -> bool:
    """Run subprocess; return True on success."""
    print(f"\n  [{label}] $ {' '.join(str(c) for c in cmd)}")
    if dry_run:
        print(f"  [{label}] DRY RUN — skipped")
        return True
    t0  = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"  [{label}] ✗  FAILED (exit {ret.returncode})  [{elapsed:.1f}s]")
        return False
    print(f"  [{label}] ✓  OK  [{elapsed:.1f}s]")
    return True


def stress_csv_path(tier: str) -> Path:
    return STRESS_DIR / f"stress_{tier}.csv"


def predict_csv_path(tier: str) -> Path:
    return PREDICT_DIR / f"flood_xgb_stress_{tier}_predictions.csv"


# ── Per-tier pipeline ──────────────────────────────────────────────────────

def run_tier(tier: str, days: int, stride: int,
             skip_viz: bool, dry_run: bool, force_alerts: bool) -> dict[str, bool]:
    label = tier.upper()
    banner(f"TIER: {label}")
    results: dict[str, bool] = {}

    # ── Step 1: stress_test_gen.py ─────────────────────────────────────
    cmd = [
        PYTHON, str(STRESS_GEN),
        "--scenario", tier,
        "--csv",
        "--days", str(days),
        "--inp", str(INP_FILE),
        "--out-dir", str(STRESS_DIR),
    ]
    ok = run(cmd, f"{label}/stress_gen", dry_run)
    results["stress_gen"] = ok
    if not ok:
        print(f"  [{label}] Aborting tier — stress_gen failed.")
        results.update({"viz": False, "predict": False, "alert": False})
        return results

    # ── Step 2: swmm_viz_pyswmm.py ────────────────────────────────────
    if not skip_viz:
        cmd = [PYTHON, str(VIZ_SCRIPT), str(INP_FILE), "--stride", str(stride)]
        ok  = run(cmd, f"{label}/viz", dry_run)
        results["viz"] = ok
        # Non-fatal — viz failure won't abort the rest
        if not ok:
            print(f"  [{label}] ⚠  viz failed — continuing pipeline.")
    else:
        print(f"\n  [{label}/viz] Skipped (--skip-viz)")
        results["viz"] = None   # None = skipped

    # ── Step 3: XGB_Predict.py ────────────────────────────────────────
    csv_in = stress_csv_path(tier)
    if not dry_run and not csv_in.exists():
        print(f"  [{label}/predict] ✗  Stress CSV not found: {csv_in}")
        results.update({"predict": False, "alert": False})
        return results

    cmd = [PYTHON, str(PREDICT), "--data", str(csv_in)]
    ok  = run(cmd, f"{label}/predict", dry_run)
    results["predict"] = ok
    if not ok:
        print(f"  [{label}] Aborting tier — predict failed.")
        results["alert"] = False
        return results

    # ── Step 4: Alert.py ──────────────────────────────────────────────
    csv_out = predict_csv_path(tier)
    if not dry_run and not csv_out.exists():
        # XGB_Predict may use a different naming convention; glob for it
        candidates = list(PREDICT_DIR.glob(f"*stress_{tier}*.csv"))
        if candidates:
            csv_out = candidates[0]
            print(f"  [{label}/alert] Using: {csv_out.name}")
        else:
            print(f"  [{label}/alert] ✗  Predict CSV not found in {PREDICT_DIR}")
            results["alert"] = False
            return results

    cmd = [PYTHON, str(ALERT), "--csv", str(csv_out)]
    
    # Add the flag to the subprocess call if requested
    if force_alerts:
        cmd.append("--force")
        
    ok  = run(cmd, f"{label}/alert", dry_run)
    results["alert"] = ok

    return results


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, dict[str, bool]]) -> None:
    banner("PIPELINE SUMMARY")
    steps = ["stress_gen", "viz", "predict", "alert"]
    header = f"  {'Tier':<10}" + "".join(f"  {s:<12}" for s in steps)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tier, res in all_results.items():
        row = f"  {tier.upper():<10}"
        for s in steps:
            v = res.get(s)
            if v is True:
                sym = "✓"
            elif v is False:
                sym = "✗"
            else:
                sym = "-"
            row += f"  {sym:<12}"
        print(row)

    any_fail = any(
        v is False
        for res in all_results.values()
        for v in res.values()
    )
    print()
    print("  RESULT:", "✗  Some steps failed." if any_fail else "✓  All steps passed.")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rapid Relay EWS — Full per-tier pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps per tier:
  1. stress_test_gen.py  --scenario <tier> --csv --days <N>
  2. swmm_viz_pyswmm.py  <inp>  --stride <stride>
  3. XGB_Predict.py      --data  data/stress_test/stress_<tier>.csv
  4. Alert.py            --csv   predictions/flood_xgb_stress_<tier>_predictions.csv

Examples:
  python simulation.py
  python simulation.py --tiers warning danger
  python simulation.py --days 7 --skip-viz --dry-run
        """,
    )
    parser.add_argument(
        "--tiers", nargs="+", default=ALL_TIERS,
        choices=ALL_TIERS, metavar="TIER",
        help="Tiers to run (default: all 4).",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Simulation days for stress_gen --csv (default: 30).",
    )
    parser.add_argument(
        "--stride", type=int, default=3,
        help="Animation frame stride for viz (default: 3).",
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip swmm_viz_pyswmm.py (fastest, no animation output).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--force-alerts", action="store_true",
        help="Force Alert.py to send notifications bypassing duplicate checks.",
    )
    # --- NEW RAMP FLAG ---
    parser.add_argument(
        "--ramp", action="store_true",
        help="Execute a sequential 'ramp up' from clear to danger, overriding --tiers selection.",
    )
    args = parser.parse_args()

    # --- RAMP LOGIC ---
    if args.ramp:
        # Force the sequence from lowest to highest severity
        args.tiers = ["all"]

    banner("Rapid Relay EWS — Pipeline Orchestrator")
    print(f"  Tiers    : {', '.join(t.upper() for t in args.tiers)}")
    print(f"  Days     : {args.days}")
    print(f"  Stride   : {args.stride}")
    print(f"  Skip viz : {args.skip_viz}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Root     : {ROOT}")

    t_total = time.time()
    all_results: dict[str, dict[str, bool]] = {}

    for tier in args.tiers:
        all_results[tier] = run_tier(
            tier      = tier,
            days      = args.days,
            stride    = args.stride,
            skip_viz  = args.skip_viz,
            dry_run   = args.dry_run,
            force_alerts = args.force_alerts,
        )

    print_summary(all_results)
    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
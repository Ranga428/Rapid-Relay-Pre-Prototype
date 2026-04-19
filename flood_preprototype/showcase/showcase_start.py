"""
showcase_start.py
=================
Showcase — Event-Driven Inference Orchestrator

HOW IT WORKS
------------
Single linear loop — no threads, no overlap:

    1. Poll Supabase row count
    2. If count unchanged  → wait --interval seconds → go to 1
    3. If count increased  → stop polling → run full pipeline
    4. Pipeline finishes   → go back to 1

Nothing runs concurrently.  The pipeline only starts after a new row is
confirmed, and polling only resumes after the pipeline fully completes.

Pipeline steps (fired on every new row):
    0a  showcase_sensor_ingest.py   ingest + calibrate → showcase_sensor.csv
    0b  showcase_proxy.py           GEE proxy → showcase_proxy.csv
    0c  showcase_merge.py           merge → showcase_merge.csv
    1   showcase_predict.py         XGB inference → showcase_predict.csv
    2   showcase_alert.py           dispatch ALL tiers (incl. CLEAR)
    3   showcase_transmission_speed log latency
    4   showcase_availability.py    recalculate → availability_report.csv

Usage
-----
    python showcase_start.py                     # poll every 1s (default)
    python showcase_start.py --interval 5        # poll every 5s
    python showcase_start.py --skip-proxy        # skip GEE fetch
    python showcase_start.py --no-post           # skip alert posting
    python showcase_start.py --no-availability   # skip availability report
"""

import os
import sys
import time
import argparse
import warnings
import traceback
from datetime import datetime, date

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import showcase_alert
from showcase_transmission_speed import log_run
from showcase_availability import generate_report

# ===========================================================================
# TERMINAL STYLE
# ===========================================================================

class C:
    _on    = sys.stdout.isatty()
    RESET  = "\033[0m"        if _on else ""
    BOLD   = "\033[1m"        if _on else ""
    DIM    = "\033[2m"        if _on else ""
    GREEN  = "\033[92m"       if _on else ""
    YELLOW = "\033[93m"       if _on else ""
    ORANGE = "\033[38;5;208m" if _on else ""
    RED    = "\033[91m"       if _on else ""
    BLUE   = "\033[94m"       if _on else ""
    CYAN   = "\033[96m"       if _on else ""
    GRAY   = "\033[90m"       if _on else ""
    WHITE  = "\033[97m"       if _on else ""

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    except Exception:
        pass

TIER_ICONS = {
    "CLEAR":   ("●", C.GREEN),
    "WATCH":   ("◆", C.YELLOW),
    "WARNING": ("▲", C.ORANGE),
    "DANGER":  ("■", C.RED),
}

ALERT_CHANNELS = ["Facebook", "Telegram", "Supabase"]

PHASE_COLORS = {
    "0a": C.BLUE,
    "0b": C.BLUE,
    "0c": C.CYAN,
    "1":  C.YELLOW,
    "2":  C.GREEN,
    "3":  C.CYAN,
    "4":  C.GRAY,
}


# ===========================================================================
# BANNER
# ===========================================================================

def print_startup_banner(interval: int, skip_proxy: bool,
                         no_post: bool, no_availability: bool) -> None:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("Rapid Relay", font="slant").rstrip()
    except Exception:
        art = "  Rapid Relay"

    print()
    for line in art.split("\n"):
        print(f"{C.CYAN}{line}{C.RESET}")
    print(f"  {C.DIM}Flood EWS — Showcase Pipeline  ·  Obando, Bulacan{C.RESET}")
    print()

    channels_str = "  +  ".join(f"{C.CYAN}{ch}{C.RESET}" for ch in ALERT_CHANNELS)

    rows = [
        ("Mode",          f"{C.GREEN}Continuous — poll → detect → run → poll{C.RESET}"),
        ("Poll interval", f"{C.CYAN}{interval}s{C.RESET}  (pipeline blocks polling while running)"),
        ("Proxy fetch",   f"{C.GRAY}disabled{C.RESET}" if skip_proxy      else f"{C.GREEN}enabled{C.RESET}"),
        ("Alert channels",channels_str),
        ("Alert tiers",   f"{C.GREEN}ALL tiers including CLEAR{C.RESET}"),
        ("Posting",       f"{C.GRAY}disabled{C.RESET}" if no_post         else f"{C.GREEN}enabled{C.RESET}"),
        ("Availability",  f"{C.GRAY}disabled{C.RESET}" if no_availability else f"{C.GREEN}saved to CSV each run{C.RESET}"),
    ]

    w = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        print(f"  {C.GRAY}{k:<{w}}{C.RESET}{v}")

    print()
    print(f"  {C.DIM}Waiting for new Supabase rows...  Press Ctrl+C to stop.{C.RESET}\n")


# ===========================================================================
# PHASE / RUN HEADERS
# ===========================================================================

def print_phase(tag: str, title: str, subtitle: str = "") -> None:
    color = PHASE_COLORS.get(tag, C.GRAY)
    sub   = f"  {C.DIM}{subtitle}{C.RESET}" if subtitle else ""
    bar   = f"{C.DIM}{'─' * 70}{C.RESET}"
    print(f"\n{bar}")
    print(f"  {C.BOLD}{color}[{tag}]{C.RESET}  {C.BOLD}{C.WHITE}{title}{C.RESET}{sub}")
    print(bar)


def print_run_header(run_n: int, delta: int, total: int) -> None:
    w  = 70
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{C.DIM}{'═' * w}{C.RESET}")
    print(
        f"  {C.BOLD}{C.WHITE}SHOWCASE RUN #{run_n}{C.RESET}  "
        f"{C.DIM}|{C.RESET}  {C.CYAN}{ts}{C.RESET}  "
        f"{C.DIM}[+{delta} row(s)  total={total}]{C.RESET}"
    )
    print(f"{C.DIM}{'═' * w}{C.RESET}")


def print_done(run_n: int) -> None:
    w = 70
    print(f"\n{C.DIM}{'═' * w}{C.RESET}")
    print(f"  {C.BOLD}{C.GREEN}✓  RUN #{run_n} DONE — resuming poll...{C.RESET}")
    print(f"{C.DIM}{'═' * w}{C.RESET}\n")


# ===========================================================================
# MODULE IMPORT HELPER
# ===========================================================================

def _import(name: str, fatal: bool = False):
    try:
        return __import__(name)
    except ImportError as e:
        if fatal:
            sys.exit(f"\n  {C.RED}ERROR{C.RESET}  Cannot import {name}: {e}\n")
        print(f"  {C.YELLOW}WARNING{C.RESET}  Cannot import {name}: {e}")
        return None


# ===========================================================================
# CSV HELPERS
# ===========================================================================

def read_latest_sensor_date() -> str | None:
    p = os.path.join(_PROJECT_ROOT, "data", "sensor", "showcase_sensor.csv")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p)
        if df.empty or "timestamp" not in df.columns:
            return None
        return df["timestamp"].astype(str).max()[:10]
    except Exception:
        return None


def read_latest_result(predict_mod) -> dict | None:
    p = predict_mod.PREDICTIONS_CSV
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
        if df.empty:
            return None
        latest = df.iloc[-1]
        return {
            "timestamp":   str(latest.name.date()),
            "probability": float(latest.get("flood_probability", 0.0)),
            "risk_tier":   str(latest.get("risk_tier", "CLEAR")),
        }
    except Exception as e:
        print(f"  {C.YELLOW}WARNING{C.RESET}  Could not read showcase_predict.csv — {e}")
        return None


# ===========================================================================
# FULL PIPELINE  (called synchronously — polling is paused while this runs)
# ===========================================================================

def run_pipeline(run_n: int, delta: int, total: int,
                 skip_proxy: bool, no_post: bool,
                 no_availability: bool) -> None:

    print_run_header(run_n, delta, total)
    ingest_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ── 0a  Sensor ingest ──────────────────────────────────────────────────
    print_phase("0a", "showcase_sensor_ingest.py",
                "Supabase pull → calibrate → showcase_sensor.csv")
    SensorIngest = _import("showcase_sensor_ingest")
    if SensorIngest:
        try:
            wrote, supabase_has_new = SensorIngest.ingest_latest()
            if wrote:
                print(f"  {C.GREEN}✓{C.RESET}  showcase_sensor.csv updated.")
            elif supabase_has_new:
                print(
                    f"  {C.DIM}No new rows to write — "
                    f"CSV unchanged, pipeline continuing.{C.RESET}"
                )
            else:
                print(f"  {C.DIM}No rows in Supabase — nothing ingested.{C.RESET}")
        except Exception as e:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Sensor ingest failed: {e}")
            traceback.print_exc()

    ingest_done = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    sensor_date = read_latest_sensor_date() or str(date.today())

    # ── 0b  Proxy fetch ────────────────────────────────────────────────────
    print_phase("0b", "showcase_proxy.py", "GEE proxy pull → showcase_proxy.csv")
    if skip_proxy:
        print(f"  {C.GRAY}Skipped.{C.RESET}")
    else:
        Proxy = _import("showcase_proxy")
        if Proxy:
            try:
                updated = Proxy.run_pipeline(force_full=False)
                if updated:
                    print(f"  {C.GREEN}✓{C.RESET}  showcase_proxy.csv updated.")
                else:
                    print(f"  {C.DIM}Already up to date.{C.RESET}")
            except Exception as e:
                print(f"  {C.YELLOW}WARNING{C.RESET}  Proxy fetch failed: {e}")
                traceback.print_exc()

    # ── 0c  Merge ──────────────────────────────────────────────────────────
    print_phase("0c", "showcase_merge.py",
                "sensor + proxy → showcase_merge.csv")
    Merge = _import("showcase_merge")
    if Merge:
        try:
            new_rows = Merge.run_pipeline()
            if new_rows is not None and len(new_rows) > 0:
                print(f"  {C.GREEN}✓{C.RESET}  Merged {len(new_rows)} row(s).")
            else:
                print(f"  {C.DIM}Already up to date.{C.RESET}")
        except Exception as e:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Merge failed: {e}")
            traceback.print_exc()

    # ── 1  Predict ─────────────────────────────────────────────────────────
    print_phase("1", "showcase_predict.py",
                "XGB inference → showcase_predict.csv")
    ShowcasePredict = _import("showcase_predict", fatal=True)
    predict_ok         = False
    predict_done_clock = None

    try:
        ShowcasePredict.run_pipeline()
        predict_ok         = True
        predict_done_clock = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        print(f"  {C.GREEN}✓{C.RESET}  Prediction complete.")
    except SystemExit as e:
        print(f"  {C.YELLOW}predict exited early:{C.RESET} {e}")
        print(f"  {C.DIM}No new rows to predict on.{C.RESET}")
    except Exception as e:
        print(f"  {C.RED}ERROR{C.RESET}  showcase_predict.py: {e}")
        traceback.print_exc()

    # ── 2  Alert ───────────────────────────────────────────────────────────
    print_phase("2", "showcase_alert.py",
                "dispatch ALL tiers → Facebook + Telegram + Supabase")
    latest = read_latest_result(ShowcasePredict) if predict_ok else None

    if latest:
        tier  = latest["risk_tier"]
        prob  = latest["probability"]
        ts    = latest["timestamp"]
        icon, color = TIER_ICONS.get(tier, ("●", C.WHITE))
        print(
            f"\n  Latest result  {C.BOLD}{color}{icon}  {tier}{C.RESET}  "
            f"{C.DIM}{prob:.1%}  ({ts}){C.RESET}"
        )
        if no_post:
            print(f"  {C.GRAY}--no-post — skipping dispatch.{C.RESET}")
        else:
            showcase_alert.dispatch_alert(tier=tier, probability=prob, timestamp=ts)
    else:
        if predict_ok:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Prediction ran but result unreadable.")
        print(f"  {C.GRAY}Skipping alert dispatch.{C.RESET}")
        tier = "CLEAR"
        prob = 0.0
        ts   = str(date.today())

    # ── 3  Transmission log ────────────────────────────────────────────────
    print_phase("3", "showcase_transmission_speed.py", "log latency")
    if predict_ok and predict_done_clock:
        try:
            prediction_date = latest["timestamp"] if latest else str(date.today())
            record = log_run(
                ingest_wall_clock  = ingest_done,
                predict_wall_clock = predict_done_clock,
                sensor_date        = sensor_date,
                prediction_date    = prediction_date,
                tier               = tier,
                probability        = prob,
            )
            if record:
                print(
                    f"  {C.GREEN}✓{C.RESET}  Latency: "
                    f"{record.get('latency_hms', 'N/A')} "
                    f"({record.get('latency_seconds', 0):.1f}s)"
                )
        except Exception as e:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Transmission log failed: {e}")
    else:
        print(f"  {C.GRAY}Skipped — no successful prediction.{C.RESET}")

    # ── 4  Availability ────────────────────────────────────────────────────
    if not no_availability:
        print_phase("4", "showcase_availability.py",
                    "recalculate uptime / downtime → availability_report.csv")
        try:
            generate_report(
                start_date=date.fromisoformat("2025-07-01"),
                save=True,
            )
        except Exception as e:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Availability report failed: {e}")

    print_done(run_n)


# ===========================================================================
# MAIN LOOP  — poll → detect → run → repeat
# ===========================================================================

def run_forever(interval: int, skip_proxy: bool,
                no_post: bool, no_availability: bool) -> None:
    """
    Linear loop.  No threads.  No overlap.

        poll Supabase row count
            unchanged  → sleep interval → poll again
            increased  → run full pipeline (blocking)
                       → poll again immediately after pipeline finishes
    """
    from showcase_sensor_ingest import get_supabase_row_count

    run_n = 1

    # Get baseline count before we start watching
    last_count = get_supabase_row_count()
    while last_count < 0:
        print(f"  {C.YELLOW}Could not reach Supabase — retrying in {interval}s...{C.RESET}")
        time.sleep(interval)
        last_count = get_supabase_row_count()

    print(
        f"  {C.GREEN}✓{C.RESET}  Connected.  "
        f"Current row count: {C.CYAN}{last_count}{C.RESET}\n"
    )

    try:
        while True:
            time.sleep(interval)

            current_count = get_supabase_row_count()

            if current_count < 0:
                ts = pd.Timestamp.now().strftime("%H:%M:%S")
                print(f"  {C.YELLOW}[{ts}] Row count unavailable — retrying next poll.{C.RESET}")
                continue

            if current_count > last_count:
                # ── New row(s) detected ─────────────────────────────────
                delta      = current_count - last_count
                last_count = current_count

                # Run the full pipeline synchronously.
                # Polling is implicitly paused — we don't call time.sleep
                # or check the count again until run_pipeline() returns.
                run_pipeline(
                    run_n           = run_n,
                    delta           = delta,
                    total           = current_count,
                    skip_proxy      = skip_proxy,
                    no_post         = no_post,
                    no_availability = no_availability,
                )
                run_n += 1
                # Loop back to top immediately — no extra sleep after pipeline

            else:
                # ── No change — print a live status line ───────────────
                ts = pd.Timestamp.now().strftime("%H:%M:%S")
                print(
                    f"\r  {C.DIM}[{ts}]  Watching...  "
                    f"rows={current_count}{C.RESET}   ",
                    end="", flush=True,
                )

    except KeyboardInterrupt:
        print(f"\n  {C.YELLOW}Stopped by user.{C.RESET}\n")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Showcase Flood EWS — Continuous event-driven orchestrator.\n"
            "Polls Supabase every --interval seconds.\n"
            "When a new row is detected: stops polling, runs the full pipeline,\n"
            "then resumes polling.  Never overlaps.  Runs until Ctrl+C."
        )
    )
    parser.add_argument(
        "--interval", type=int, default=1,
        help="Seconds between each Supabase row-count poll (default: 1).",
    )
    parser.add_argument(
        "--skip-proxy", action="store_true",
        help="Skip GEE proxy fetch on every run.",
    )
    parser.add_argument(
        "--no-post", action="store_true",
        help="Skip alert posting to Facebook / Telegram / Supabase.",
    )
    parser.add_argument(
        "--no-availability", action="store_true",
        help="Skip availability report recalculation after each run.",
    )
    args = parser.parse_args()

    print_startup_banner(
        interval        = args.interval,
        skip_proxy      = args.skip_proxy,
        no_post         = args.no_post,
        no_availability = args.no_availability,
    )

    run_forever(
        interval        = args.interval,
        skip_proxy      = args.skip_proxy,
        no_post         = args.no_post,
        no_availability = args.no_availability,
    )
"""
Start.py
========
Flood Prediction — Daily Inference Orchestrator

PURPOSE
-------
Single entry point for daily operational deployment.
Runs the full pipeline in order, all incremental:

    Step 0a  sensor_ingest.py         — pull new hardware readings from Supabase
    Step 0b  Sat_SensorData_proxy.py  — pull new proxy data from GEE
    Step 0c  merge_sensor.py          — merge both into combined_sensor_context.csv
    Step 1   XGB_Predict.py           — live mode inference on combined CSV
    Step 2   RF_Predict.py            — comparison run (if --all-models)
    Step 3   LGBM_Predict.py          — comparison run (if --all-models)
    Step 4   Alert.py                 — dispatch alert (Facebook + Telegram)
    Step 5   Seasonal notification    — remind to retrain every ~90 days (manual trigger)

ALL STEPS ARE INCREMENTAL
--------------------------
Every script only processes rows newer than what is already on disk:
    sensor_ingest   — fetches Supabase rows newer than latest obando_sensor_data.csv timestamp
    Sat_SensorData  — fetches GEE rows newer than latest obando_environmental_data.csv timestamp
    merge_sensor    — merges only rows newer than latest combined_sensor_context.csv timestamp
    XGB_Predict     — predicts only rows after LAST_TRAINING_DATE (filter_new_rows)
    Alert           — FB + TG check_duplicate=True; Supabase appends only new prediction rows

ALERT DISPATCH
--------------
Alert.py always runs last, after all model predictions are complete.
It reads the latest row from flood_xgb_sensor_predictions.csv — the XGB
model is the primary operational model and is always the alert source.
RF and LGBM are comparison-only and do not trigger alerts.

Alerts are dispatched to two channels:
    • Facebook Page  (Alert.py → FB_Alert.py)
    • Telegram Group (Alert.py → TG_Alert.py)

SCHEDULER — MIDNIGHT-PINNED
-----------------------------
    python Start.py --schedule

Runs once immediately on start, then sleeps until the next calendar midnight
(local system time) before each subsequent run. This means the pipeline always
fires at midnight regardless of when you first started the script — not every
24 hours from launch time.

If a run takes longer than expected and the next midnight has already passed,
the script runs immediately and re-aligns to the following midnight.

RETRAINING (manual, separate script)
--------------------------------------
    python Retraining_Pipeline.py --retrain

NOT run automatically. Start.py checks days since last training and prints
a prominent terminal notice when retraining is due (~every 90 days).

Usage
-----
    # Run once manually
    python Start.py

    # Run on midnight schedule (keeps running, Ctrl+C to stop)
    python Start.py --schedule

    # Skip hardware sensor ingest (no Supabase connection / testing)
    python Start.py --skip-sensor

    # Skip GEE proxy fetch (no GEE auth / testing)
    python Start.py --skip-proxy

    # Skip all alert posting (dry run / testing)
    python Start.py --no-post

    # Full scheduled run
    python Start.py --schedule

    # Also run RF and LGBM for comparison logging
    python Start.py --all-models
"""

import os
import sys
import time
import argparse
import warnings
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "ml_pipeline"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "alerts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "deployment"))

import Alert


# ===========================================================================
# CONFIG
# ===========================================================================

RETRAIN_NOTIFY_DAYS = 90

LOG_DIR     = os.path.join(SCRIPT_DIR, "..", "logs")
RETRAIN_LOG = os.path.join(LOG_DIR, "retrain_notifications.log")

# Alert channels — displayed in the banner
ALERT_CHANNELS = ["Facebook", "Telegram"]


# ===========================================================================
# TERMINAL STYLE  (ported from Rapid_Relay_Showcase.py)
# ===========================================================================

class C:
    """ANSI color codes. Auto-disabled if not a TTY."""
    _on = sys.stdout.isatty()
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

# Enable ANSI escape codes on Windows (no-op on macOS/Linux)
if sys.platform == "win32":
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

PHASE_COLORS = {
    "0a": C.BLUE,
    "0b": C.BLUE,
    "0c": C.CYAN,
    "1":  C.YELLOW,
    "2a": C.YELLOW,
    "2b": C.YELLOW,
    "3":  C.RED,
    "4":  C.GREEN,
}

TIER_ICONS = {
    "CLEAR":   ("●", C.GREEN),
    "WATCH":   ("◆", C.YELLOW),
    "WARNING": ("▲", C.ORANGE),
    "DANGER":  ("■", C.RED),
}


def _short(path: str) -> str:
    p = Path(path)
    try:
        rel   = p.relative_to(Path.cwd())
        parts = rel.parts
    except ValueError:
        parts = p.parts
    if len(parts) == 0:
        return path
    if len(parts) == 1:
        return parts[0]
    slug = f"…/{'/'.join(parts[-2:])}" if len(parts) > 2 else str(Path(*parts))
    return slug[:42] + "…" if len(slug) > 42 else slug


def _link(full: str, label: str | None = None) -> str:
    """OSC 8 hyperlink — short display label, full path on hover/copy."""
    if not sys.stdout.isatty():
        return full
    short = label or _short(full)
    return f"\033]8;;{full}\033\\{short}\033]8;;\033\\"


# ===========================================================================
# BANNER
# ===========================================================================

def print_startup_banner(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
    scheduled:      bool = False,
) -> None:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("Rapid Relay", font="slant").rstrip()
    except Exception:
        art = "  Rapid Relay"

    print()
    for line in art.split("\n"):
        print(f"{C.CYAN}{line}{C.RESET}")
    print(f"  {C.DIM}Flood Early Warning System  ·  Obando, Bulacan{C.RESET}")
    print()

    channels_str = "  +  ".join(
        f"{C.CYAN}{ch}{C.RESET}" for ch in ALERT_CHANNELS
    )

    rows = [
        ("Mode",           f"{C.GREEN}Scheduled  (midnight-pinned){C.RESET}" if scheduled else f"{C.YELLOW}Single run{C.RESET}"),
        ("Primary model",  f"XGB sensor  {C.DIM}(live, incremental){C.RESET}"),
        ("Comparison",     f"RF + LGBM" if run_all_models else f"{C.DIM}XGB only{C.RESET}"),
        ("Hardware ingest",f"{C.GRAY}disabled  (--skip-sensor){C.RESET}" if skip_sensor else f"{C.GREEN}enabled{C.RESET}  {C.DIM}Supabase incremental{C.RESET}"),
        ("Proxy fetch",    f"{C.GRAY}disabled  (--skip-proxy){C.RESET}"  if skip_proxy  else f"{C.GREEN}enabled{C.RESET}  {C.DIM}GEE incremental{C.RESET}"),
        ("Alert channels", channels_str),
        ("Posting",        f"{C.GRAY}disabled  (--no-post){C.RESET}" if no_post else f"{C.GREEN}enabled{C.RESET}  {C.DIM}WATCH / WARNING / DANGER{C.RESET}"),
        ("Alert source",   f"flood_xgb_sensor_predictions.csv  {C.DIM}(always XGB){C.RESET}"),
    ]

    w = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        print(f"  {C.GRAY}{k:<{w}}{C.RESET}{v}")

    print()
    print(f"  {C.DIM}Press Ctrl+C to stop.{C.RESET}\n")


# ===========================================================================
# PHASE HEADERS
# ===========================================================================

def print_phase(tag: str, title: str, subtitle: str = "") -> None:
    color = PHASE_COLORS.get(tag, C.GRAY)
    sub   = f"  {C.DIM}{subtitle}{C.RESET}" if subtitle else ""
    bar   = f"{C.DIM}{'─' * 70}{C.RESET}"
    print(f"\n{bar}")
    print(f"  {C.BOLD}{color}[{tag}]{C.RESET}  {C.BOLD}{C.WHITE}{title}{C.RESET}{sub}")
    print(bar)


def print_run_header(run_n: int | None = None) -> None:
    w  = 70
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{C.DIM}{'═' * w}{C.RESET}")
    if run_n is not None:
        print(f"  {C.BOLD}{C.WHITE}RUN #{run_n}{C.RESET}  {C.DIM}|{C.RESET}  {C.CYAN}{ts}{C.RESET}")
    else:
        print(f"  {C.BOLD}{C.WHITE}Flood EWS — Daily Run{C.RESET}  {C.DIM}|{C.RESET}  {C.CYAN}{ts}{C.RESET}")
    print(f"{C.DIM}{'═' * w}{C.RESET}")


def print_result_block(tier: str, prob: float, ts: str, xgb_csv: str, xgb_plot: str) -> None:
    icon, color = TIER_ICONS.get(tier, ("●", C.WHITE))
    w = 70
    print(f"\n{C.DIM}{'─' * w}{C.RESET}")
    print(
        f"  {C.BOLD}{color}{icon}  {tier}{C.RESET}"
        f"  {C.DIM}flood probability{C.RESET}"
        f"  {C.BOLD}{C.WHITE}{prob:.1%}{C.RESET}"
        f"  {C.DIM}({ts}){C.RESET}"
    )
    print(f"{C.DIM}{'─' * w}{C.RESET}")
    fields = [
        ("Predictions CSV", _link(xgb_csv)),
        ("Plot",            _link(xgb_plot)),
        ("Channels",        "  +  ".join(ALERT_CHANNELS)),
    ]
    for k, v in fields:
        print(f"  {C.GRAY}{k:<18}{C.RESET}{C.BLUE}{v}{C.RESET}")


def print_done() -> None:
    w = 70
    print(f"\n{C.DIM}{'═' * w}{C.RESET}")
    print(f"  {C.BOLD}{C.GREEN}✓  DONE{C.RESET}")
    print(f"{C.DIM}{'═' * w}{C.RESET}\n")


# ===========================================================================
# IMPORT HELPERS
# ===========================================================================

def import_sensor_ingest():
    try:
        import sensor_ingest
        return sensor_ingest
    except ImportError as e:
        print(f"  {C.YELLOW}WARNING{C.RESET}  Could not import sensor_ingest.py — {e}")
        print(f"  {C.DIM}Hardware sensor CSV will NOT be updated.{C.RESET}")
        return None


def import_proxy_module():
    try:
        import Sat_SensorData_proxy
        return Sat_SensorData_proxy
    except ImportError as e:
        print(f"  {C.YELLOW}WARNING{C.RESET}  Could not import Sat_SensorData_proxy.py — {e}")
        print(f"  {C.DIM}Proxy (satellite) CSV will NOT be updated.{C.RESET}")
        return None


def import_merge_module():
    try:
        import merge_sensor
        return merge_sensor
    except ImportError as e:
        print(f"  {C.YELLOW}WARNING{C.RESET}  Could not import merge_sensor.py — {e}")
        print(f"  {C.DIM}combined_sensor_context.csv will NOT be updated.{C.RESET}")
        return None


def import_predict_module():
    try:
        import XGB_Predict
        return XGB_Predict
    except ImportError as e:
        sys.exit(
            f"\n  {C.RED}ERROR{C.RESET}  Could not import XGB_Predict.py.\n"
            f"  Make sure it is in the same folder as Start.py.\n"
            f"  Detail: {e}\n"
        )


def import_comparison_modules():
    modules = {}
    for name in ("RF_Predict", "LGBM_Predict"):
        try:
            mod = __import__(name)
            modules[name] = mod
            print(f"  {C.GREEN}✓{C.RESET}  Loaded {name}.py")
        except ImportError as e:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Could not import {name}.py — skipping. ({e})")
    return modules


# ===========================================================================
# RETRAIN NOTIFICATION
# ===========================================================================

def check_retrain_notification(XGB_Predict) -> None:
    try:
        import joblib
        artifact   = joblib.load(XGB_Predict.MODEL_FILE)
        last_train = artifact.get("last_training_date")
        if last_train is None:
            return

        last_train_date = pd.Timestamp(last_train).date()
        days_elapsed    = (date.today() - last_train_date).days
        months_elapsed  = days_elapsed / 30.0

        if days_elapsed >= RETRAIN_NOTIFY_DAYS:
            w = 70
            print(f"\n{C.DIM}{'─' * w}{C.RESET}")
            print(f"  {C.BOLD}{C.YELLOW}⚠  SEASONAL RETRAIN REMINDER{C.RESET}")
            print(f"{C.DIM}{'─' * w}{C.RESET}")
            print(f"  {C.GRAY}Last training   {C.RESET}{last_train_date}")
            print(f"  {C.GRAY}Days elapsed    {C.RESET}{C.YELLOW}{days_elapsed}{C.RESET}  {C.DIM}(~{months_elapsed:.1f} months){C.RESET}")
            print(f"  {C.GRAY}Threshold       {C.RESET}{RETRAIN_NOTIFY_DAYS} days")
            print()
            print(f"  The model has not been retrained this season.")
            print(f"  New Sentinel-1 passes and sensor readings have accumulated.")
            print(f"  Retraining is recommended to incorporate recent flood patterns.")
            print()
            print(f"  {C.DIM}To retrain (manual step):{C.RESET}")
            print(f"  {C.CYAN}python Retraining_Pipeline.py --retrain{C.RESET}")
            print(f"{C.DIM}{'─' * w}{C.RESET}")

            os.makedirs(LOG_DIR, exist_ok=True)
            try:
                with open(RETRAIN_LOG, "a") as f:
                    f.write(
                        f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}] "
                        f"Retrain notification fired — last_training_date={last_train_date}, "
                        f"days_elapsed={days_elapsed}\n"
                    )
            except Exception:
                pass
        else:
            days_remaining = RETRAIN_NOTIFY_DAYS - days_elapsed
            print(
                f"  {C.GRAY}Retrain check   {C.RESET}"
                f"{days_elapsed}d since last training  "
                f"{C.DIM}({days_remaining}d until seasonal reminder){C.RESET}"
            )

    except Exception as e:
        print(f"  {C.YELLOW}NOTE{C.RESET}  Could not check retrain status — {e}")


# ===========================================================================
# READ LATEST PREDICTION
# ===========================================================================

def read_latest_result(XGB_Predict) -> dict | None:
    csv_path = XGB_Predict.PREDICTIONS_CSV
    if not os.path.exists(csv_path):
        print(f"  {C.YELLOW}WARNING{C.RESET}  Predictions CSV not found at {_link(csv_path)}")
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        if len(df) == 0:
            return None
        latest = df.iloc[-1]
        return {
            "timestamp":   str(latest.name.date()),
            "probability": float(latest.get("flood_probability", 0.0)),
            "risk_tier":   str(latest.get("risk_tier", "CLEAR")),
        }
    except Exception as e:
        print(f"  {C.YELLOW}WARNING{C.RESET}  Could not read predictions CSV — {e}")
        return None


# ===========================================================================
# MIDNIGHT SLEEP
# ===========================================================================

def seconds_until_midnight() -> float:
    now      = datetime.now()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max((tomorrow - now).total_seconds(), 1.0)


# ===========================================================================
# MAIN DAILY RUN
# ===========================================================================

def run_daily(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
    run_n:          int | None = None,
) -> None:

    print_run_header(run_n)

    # ── Step 0a: Hardware sensor ingest ────────────────────────────────────
    print_phase("0a", "sensor_ingest.py", "Supabase hardware pull")
    if skip_sensor:
        print(f"  {C.GRAY}Skipped — --skip-sensor flag set.{C.RESET}")
    else:
        SensorIngest = import_sensor_ingest()
        if SensorIngest is not None:
            try:
                updated = SensorIngest.ingest_latest()
                if updated:
                    print(f"  {C.GREEN}✓{C.RESET}  Hardware sensor CSV updated with latest Supabase rows.")
                else:
                    print(f"  {C.DIM}Hardware sensor CSV already up to date — no new rows.{C.RESET}")
            except Exception as e:
                print(f"  {C.YELLOW}WARNING{C.RESET}  Hardware ingest failed: {e}")
                print(f"  {C.DIM}Continuing with existing obando_sensor_data.csv.{C.RESET}")
                traceback.print_exc()
        else:
            print(f"  {C.GRAY}Skipped — sensor_ingest module not available.{C.RESET}")

    # ── Step 0b: GEE proxy fetch ────────────────────────────────────────────
    print_phase("0b", "Sat_SensorData_proxy.py", "GEE proxy pull")
    if skip_proxy:
        print(f"  {C.GRAY}Skipped — --skip-proxy flag set.{C.RESET}")
    else:
        Proxy = import_proxy_module()
        if Proxy is not None:
            try:
                updated = Proxy.run_pipeline(force_full=False)
                if updated:
                    print(f"  {C.GREEN}✓{C.RESET}  Proxy CSV updated with latest GEE data.")
                else:
                    print(f"  {C.DIM}Proxy CSV already up to date — no new rows.{C.RESET}")
            except Exception as e:
                print(f"  {C.YELLOW}WARNING{C.RESET}  Proxy fetch failed: {e}")
                print(f"  {C.DIM}Continuing with existing obando_environmental_data.csv.{C.RESET}")
                traceback.print_exc()
        else:
            print(f"  {C.GRAY}Skipped — Sat_SensorData_proxy module not available.{C.RESET}")

    # ── Step 0c: Merge ──────────────────────────────────────────────────────
    print_phase("0c", "merge_sensor.py", "merge proxy + hardware → combined_sensor_context.csv")
    Merge = import_merge_module()
    if Merge is not None:
        try:
            new_rows = Merge.run_pipeline()
            if new_rows is not None and len(new_rows) > 0:
                print(f"  {C.GREEN}✓{C.RESET}  Merged {len(new_rows)} new row(s) into combined_sensor_context.csv.")
            else:
                print(f"  {C.DIM}combined_sensor_context.csv already up to date — no new rows.{C.RESET}")
        except Exception as e:
            print(f"  {C.YELLOW}WARNING{C.RESET}  Merge failed: {e}")
            print(f"  {C.DIM}XGB_Predict will use whatever combined_sensor_context.csv currently exists.{C.RESET}")
            traceback.print_exc()
    else:
        print(f"  {C.GRAY}Skipped — merge_sensor module not available.{C.RESET}")
        print(f"  {C.DIM}XGB_Predict will fall back to obando_environmental_data.csv (proxy only).{C.RESET}")

    # ── Step 1: XGB prediction ──────────────────────────────────────────────
    print_phase("1", "XGB_Predict.py", "live mode — incremental")
    XGB_Predict = import_predict_module()
    xgb_ok = False
    try:
        XGB_Predict.run_pipeline()
        xgb_ok = True
        print(f"  {C.GREEN}✓{C.RESET}  XGB pipeline complete.")
    except SystemExit as e:
        print(f"  {C.YELLOW}XGB_Predict exited early:{C.RESET} {e}")
        print(f"  {C.DIM}Most likely: no new sensor rows after the training cutoff.{C.RESET}")
    except Exception as e:
        print(f"  {C.RED}ERROR{C.RESET}  XGB_Predict.py: {e}")
        traceback.print_exc()

    # ── Step 2: Comparison models (optional) ────────────────────────────────
    if run_all_models:
        comparison_mods = import_comparison_modules()

        if "RF_Predict" in comparison_mods:
            print_phase("2a", "RF_Predict.py", "comparison run")
            try:
                comparison_mods["RF_Predict"].run_pipeline()
                print(f"  {C.GREEN}✓{C.RESET}  RF pipeline complete.")
            except SystemExit:
                print(f"  {C.DIM}RF: no new rows or early exit.{C.RESET}")
            except Exception as e:
                print(f"  {C.RED}ERROR{C.RESET}  RF_Predict.py: {e}")

        if "LGBM_Predict" in comparison_mods:
            print_phase("2b", "LGBM_Predict.py", "comparison run")
            try:
                comparison_mods["LGBM_Predict"].run_pipeline()
                print(f"  {C.GREEN}✓{C.RESET}  LGBM pipeline complete.")
            except SystemExit:
                print(f"  {C.DIM}LGBM: no new rows or early exit.{C.RESET}")
            except Exception as e:
                print(f"  {C.RED}ERROR{C.RESET}  LGBM_Predict.py: {e}")

    # ── Step 3: Alert dispatch — Facebook + Telegram ─────────────────────────
    #
    # Always reads flood_xgb_sensor_predictions.csv.
    # Both FB and Telegram fire through Alert.py; check_duplicate=True on each
    # channel so a re-run on the same day won't double-post.
    #
    print_phase("3", "Alert.py", "dispatch → Facebook  +  Telegram")
    latest = read_latest_result(XGB_Predict) if xgb_ok else None

    if latest:
        tier  = latest["risk_tier"]
        prob  = latest["probability"]
        ts    = latest["timestamp"]
        icon, color = TIER_ICONS.get(tier, ("●", C.WHITE))

        print(
            f"\n  Latest XGB result   "
            f"{C.BOLD}{color}{icon}  {tier}{C.RESET}  "
            f"{C.DIM}{prob:.1%}  ({ts}){C.RESET}"
        )
        print(
            f"  Alert channels      "
            f"{C.CYAN}Facebook{C.RESET}  {C.DIM}+{C.RESET}  {C.CYAN}Telegram{C.RESET}"
        )

        if no_post:
            print(f"\n  {C.GRAY}--no-post flag set — skipping alert dispatch.{C.RESET}")
        else:
            Alert.dispatch_alert(tier=tier, probability=prob, timestamp=ts)

        # Summary block
        print_result_block(
            tier     = tier,
            prob     = prob,
            ts       = ts,
            xgb_csv  = XGB_Predict.PREDICTIONS_CSV,
            xgb_plot = XGB_Predict.PLOT_FILE,
        )
    else:
        if xgb_ok:
            print(f"\n  {C.YELLOW}WARNING{C.RESET}  XGB ran but could not read back results.")
        print(f"  {C.GRAY}Skipping alert dispatch.{C.RESET}")

    # ── Step 4: Seasonal retrain check ─────────────────────────────────────
    print_phase("4", "Seasonal Retrain Check")
    check_retrain_notification(XGB_Predict)

    print_done()


# ===========================================================================
# SCHEDULED MODE
# ===========================================================================

def run_scheduled(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
) -> None:

    run_n = 1

    while True:
        try:
            run_daily(
                skip_sensor    = skip_sensor,
                skip_proxy     = skip_proxy,
                no_post        = no_post,
                run_all_models = run_all_models,
                run_n          = run_n,
            )
            run_n += 1

            total_secs = seconds_until_midnight()
            next_run   = datetime.now() + timedelta(seconds=total_secs)
            next_str   = next_run.strftime("%Y-%m-%d 00:00")
            print(f"  {C.GRAY}Next run{C.RESET}  {C.CYAN}{next_str}{C.RESET}")

            remaining = total_secs
            try:
                while remaining > 0:
                    h, r = divmod(int(remaining), 3600)
                    m, s = divmod(r, 60)
                    print(
                        f"\r  {C.DIM}⏳  Sleeping: {C.CYAN}{h:02d}:{m:02d}:{s:02d}{C.RESET}{C.DIM} until midnight{C.RESET}   ",
                        end="", flush=True,
                    )
                    time.sleep(1)
                    remaining -= 1
                print(f"\r{' ' * 60}\r", end="", flush=True)
                print(f"  {C.GREEN}✓{C.RESET}  Starting next run...", flush=True)
                time.sleep(0.3)
                print(f"\r{' ' * 55}\r", end="", flush=True)
            except KeyboardInterrupt:
                print(f"\n  {C.YELLOW}Stopped by user.{C.RESET}\n")
                return

        except KeyboardInterrupt:
            print(f"\n  {C.YELLOW}Stopped by user.{C.RESET}\n")
            break
        except Exception as exc:
            print(f"\n  {C.RED}Unexpected error:{C.RESET} {exc}")
            traceback.print_exc()
            secs = seconds_until_midnight()
            print(f"  {C.DIM}Retrying at next midnight...{C.RESET}")
            time.sleep(secs)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flood Early Warning System — Daily Inference Orchestrator.\n"
            "Steps: hardware ingest → proxy fetch → merge → XGB predict → "
            "RF/LGBM (optional) → alert (Facebook + Telegram) → retrain check.\n"
            "All steps are incremental — only new rows processed each run.\n"
            "Alert always dispatches from flood_xgb_sensor_predictions.csv."
        )
    )
    parser.add_argument("--schedule",     action="store_true",
                        help="Run on a midnight-pinned daily schedule.")
    parser.add_argument("--skip-sensor",  action="store_true",
                        help="Skip Step 0a — Supabase hardware ingest.")
    parser.add_argument("--skip-proxy",   action="store_true",
                        help="Skip Step 0b — GEE proxy fetch.")
    parser.add_argument("--no-post",      action="store_true",
                        help="Skip all alert dispatch (Facebook + Telegram).")
    parser.add_argument("--all-models",   action="store_true",
                        help="Also run RF_Predict.py and LGBM_Predict.py for comparison.")
    args = parser.parse_args()

    print_startup_banner(
        skip_sensor    = args.skip_sensor,
        skip_proxy     = args.skip_proxy,
        no_post        = args.no_post,
        run_all_models = args.all_models,
        scheduled      = args.schedule,
    )

    if args.schedule:
        run_scheduled(
            skip_sensor    = args.skip_sensor,
            skip_proxy     = args.skip_proxy,
            no_post        = args.no_post,
            run_all_models = args.all_models,
        )
    else:
        run_daily(
            skip_sensor    = args.skip_sensor,
            skip_proxy     = args.skip_proxy,
            no_post        = args.no_post,
            run_all_models = args.all_models,
        )
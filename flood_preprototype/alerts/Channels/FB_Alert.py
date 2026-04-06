"""
FB_Alert.py
===========
Flood EWS — Facebook Page Alert Channel

Posts flood alerts to the configured Facebook Page via the Graph API.

CREDENTIALS
-----------
Reads from .env located TWO levels above this file (project root):
    FB_PAGE_ID      — numeric Facebook Page ID
    FB_PAGE_TOKEN   — long-lived Page access token

    .env location: ../../.env  relative to this file
    i.e. D:/Rapid-Relay/Rapid-Relay-Pre-Prototype/flood_preprototype/../../.env

    Example .env entries:
        FB_PAGE_ID=123456789012345
        FB_PAGE_TOKEN=EAAxxxxxxxxxxxxxxx...

TOKEN EXPIRY
------------
Long-lived page tokens expire after ~60 days. Renew at:
    https://developers.facebook.com/tools/explorer/
When renewed, update FB_PAGE_TOKEN in .env — no code changes needed.

GRAPH API VERSION
-----------------
Currently using v23.0 — matches the working test script.
Update the VERSION constant below if the API version changes.

Usage
-----
    # Called by Alert.py — not run directly in production
    from alerts.FB_Alert import send

    ok = send(tier="WARNING", probability=0.52, timestamp="2025-07-14")

    # Post from latest CSV row (real data, skips if already posted)
    python FB_Alert.py --post

    # Test manually with fake data (no duplicate check)
    python FB_Alert.py --test
    python FB_Alert.py --test --tier DANGER

ANTI-DUPLICATE
--------------
--post writes the posted timestamp to last_posted.json (same directory).
Re-running --post with the same CSV row is a safe no-op — it will log
⏭️  Skipped and exit without calling the API.
--test bypasses this check entirely.
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# .env is two levels above this file: alerts/ → flood_preprototype/ → project root
_ENV_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

PAGE_ID = os.getenv("FB_PAGE_ID")
TOKEN   = os.getenv("FB_PAGE_TOKEN")

VERSION = "v23.0"

# CSV is two levels up from this file, inside predictions/
CSV_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR, "..", "..",
        "predictions",
        "flood_rf_sensor_predictions.csv"
    )
)

# Tracks the last successfully posted timestamp to prevent duplicate posts
STATE_PATH = os.path.join(SCRIPT_DIR, "last_posted.json")

# Tiers that carry no flood risk — skip API call, terminal notice only
SKIP_TIERS = {"CLEAR", "LOW", "NORMAL"}

TIER_EMOJI = {
    "WATCH":   "🟡",
    "WARNING": "🟠",
    "DANGER":  "🔴",
}

TIER_MESSAGE = {
    "WATCH": (
        "🟡 FLOOD WATCH — Obando, Bulacan\n\n"
        "Elevated flood risk detected by the Rapid Relay Early Warning System.\n"
        "Monitor water levels closely. Be prepared to act.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "Stay safe. Follow official advisories from local DRRMO."
    ),
    "WARNING": (
        "🟠 FLOOD WARNING — Obando, Bulacan\n\n"
        "High flood risk detected by the Rapid Relay Early Warning System.\n"
        "Prepare for possible flooding. Move valuables to higher ground.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "Stay safe. Follow official advisories from local DRRMO."
    ),
    "DANGER": (
        "🔴 FLOOD DANGER — Obando, Bulacan\n\n"
        "IMMINENT or ONGOING flood detected by the Rapid Relay Early Warning System.\n"
        "Take immediate action. Evacuate if in flood-prone areas.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "🚨 Follow evacuation orders from local DRRMO immediately."
    ),
}


# Tiers that carry no flood risk — skip posting entirely
SKIP_TIERS = {"CLEAR", "LOW", "NORMAL"}


def _load_last_posted() -> str | None:
    """
    Read last_posted.json and return the last posted timestamp string.
    Returns None if the file doesn't exist or is unreadable.
    """
    try:
        if not os.path.exists(STATE_PATH):
            return None
        with open(STATE_PATH, "r") as f:
            import json
            data = json.load(f)
            return data.get("last_posted_timestamp")
    except Exception as e:
        print(f"  ⚠️  Could not read state file: {e}")
        return None


def _save_last_posted(timestamp: str) -> None:
    """Write the successfully posted timestamp to last_posted.json."""
    try:
        import json
        with open(STATE_PATH, "w") as f:
            json.dump({"last_posted_timestamp": timestamp}, f, indent=2)
        print(f"  💾 State saved — last posted: {timestamp}")
    except Exception as e:
        print(f"  ⚠️  Could not save state file: {e}")


def _check_credentials() -> bool:
    """Verify PAGE_ID and TOKEN are loaded. Print helpful error if not."""
    if not PAGE_ID or not TOKEN:
        print("  ❌ Missing credentials.")
        print(f"  📂 .env path checked: {os.path.abspath(_ENV_PATH)}")
        print("  ℹ️  Ensure FB_PAGE_ID and FB_PAGE_TOKEN are set in .env")
        return False
    return True


def build_message(tier: str, probability: float, timestamp: str) -> str:
    """Build the Facebook post message string for the given tier."""
    template = TIER_MESSAGE.get(tier, TIER_MESSAGE["WARNING"])
    return template.format(
        prob_pct  = f"{probability:.1%}",
        timestamp = timestamp,
    )


def read_latest_from_csv() -> dict | None:
    """
    Read the last row (by timestamp) from the predictions CSV.

    Returns a dict with keys: tier, probability, timestamp
    Returns None if the CSV is missing, empty, or malformed.
    """
    print(f"  📂 CSV path: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print("  ❌ CSV not found.")
        return None

    try:
        df = pd.read_csv(CSV_PATH)

        if df.empty:
            print("  ❌ CSV is empty.")
            return None

        required_cols = ["timestamp", "flood_probability", "risk_tier"]
        for col in required_cols:
            if col not in df.columns:
                print(f"  ❌ Missing column: {col}")
                return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        latest = df.sort_values("timestamp").iloc[-1]

        row = {
            "tier":        str(latest["risk_tier"]).upper(),
            "probability": float(latest["flood_probability"]),
            "timestamp":   latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
        }

        print(f"  ℹ️  Latest row — tier: {row['tier']} | prob: {row['probability']:.1%} | ts: {row['timestamp']}")
        return row

    except Exception as e:
        print(f"  ❌ Failed to read CSV: {e}")
        return None


def send(tier: str, probability: float, timestamp: str, check_duplicate: bool = False) -> bool:
    """
    Post a flood alert to the configured Facebook Page.

    Parameters
    ----------
    tier             : "WATCH", "WARNING", or "DANGER"
    probability      : float 0.0–1.0
    timestamp        : date string e.g. "2025-07-14"
    check_duplicate  : if True, skips posting if timestamp matches last_posted.json

    Returns True on success (or skip), False on any failure.
    """
    if not _check_credentials():
        return False

    # Non-alert tiers — log and skip without calling the API
    if tier in SKIP_TIERS:
        print(f"  ⏭️  No post needed — tier is {tier} (prob: {probability:.1%}, ts: {timestamp})")
        _save_last_posted(timestamp)
        return True

    if check_duplicate:
        last = _load_last_posted()
        if last and last == timestamp:
            print(f"  ⏭️  Skipped — timestamp already posted: {timestamp}")
            return True

    message = build_message(tier, probability, timestamp)
    url     = f"https://graph.facebook.com/{VERSION}/{PAGE_ID}/feed"

    print(f"  🚀 Sending {tier} alert to Facebook Page")

    try:
        res = requests.post(
            url,
            data={
                "message":      message,
                "access_token": TOKEN,
            },
            timeout=15,
        )
        data = res.json()

        if "id" in data:
            print(f"  ✅ Posted successfully — post id: {data['id']}")
            _save_last_posted(timestamp)
            return True
        else:
            error = data.get("error", {})
            print(f"  ❌ API error: {error.get('message', data)}")
            return False

    except requests.Timeout:
        print("  ❌ Request timed out.")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Facebook alert channel.")
    parser.add_argument("--post", action="store_true", help="Post alert from latest CSV row (real data).")
    parser.add_argument("--test", action="store_true", help="Send a test post with fake data.")
    parser.add_argument(
        "--tier", default="WARNING",
        choices=["WATCH", "WARNING", "DANGER"],
        help="Tier to use with --test (default: WARNING).",
    )
    args = parser.parse_args()

    if args.post:
        print(f"\n  🚀 FB_Alert --post — reading latest CSV row")
        print(f"  ℹ️  PAGE_ID loaded : {'✅' if PAGE_ID else '❌ MISSING'}")
        print(f"  ℹ️  TOKEN loaded   : {'✅' if TOKEN else '❌ MISSING'}")
        row = read_latest_from_csv()
        if row is None:
            print("\n  ❌ FAILED — could not read CSV.")
        else:
            ok = send(
                tier=row["tier"],
                probability=row["probability"],
                timestamp=row["timestamp"],
                check_duplicate=True,
            )
            print(f"\n  {'✅ SUCCESS' if ok else '❌ FAILED'}")

    elif args.test:
        from datetime import date
        print(f"\n  🚀 FB_Alert --test — tier={args.tier}")
        print(f"  ℹ️  PAGE_ID loaded : {'✅' if PAGE_ID else '❌ MISSING'}")
        print(f"  ℹ️  TOKEN loaded   : {'✅' if TOKEN else '❌ MISSING'}")
        print(f"  📂 .env resolved  : {os.path.abspath(_ENV_PATH)}")
        ok = send(tier=args.tier, probability=0.55, timestamp=str(date.today()))
        print(f"\n  {'✅ SUCCESS' if ok else '❌ FAILED'}")

    else:
        parser.print_help()
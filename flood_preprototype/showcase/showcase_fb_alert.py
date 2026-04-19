"""
showcase_fb_alert.py
====================
SHOWCASE VERSION of FB_Alert.py

Changes from original:
  - Reads from  : showcase_predict.csv  (instead of flood_rf_sensor_predictions.csv)
  - State file  : showcase_last_fb_posted.json
  - CLEAR tier  : POSTED (not skipped) — showcase dispatches all tiers
  - All Graph API logic is identical to the original.

Usage
-----
    from showcase_fb_alert import send, read_latest_from_csv
    python showcase_fb_alert.py --post
    python showcase_fb_alert.py --test --tier CLEAR
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# .env is three levels up: showcase/ → flood_preprototype/ → Rapid-Relay-Pre-Prototype/ → project root
_ENV_PATH = os.path.join(SCRIPT_DIR, "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

PAGE_ID = os.getenv("FB_PAGE_ID")
TOKEN   = os.getenv("FB_PAGE_TOKEN")

VERSION = "v23.0"

# Reads showcase_predict.csv (in predictions/ folder)
CSV_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "predictions", "showcase_predict.csv")
)

STATE_PATH = os.path.join(SCRIPT_DIR, "showcase_last_fb_posted.json")

# Showcase: CLEAR tier is posted (not skipped)
SKIP_TIERS: set = set()

TIER_EMOJI = {
    "CLEAR":   "🟢",
    "WATCH":   "🟡",
    "WARNING": "🟠",
    "DANGER":  "🔴",
}

TIER_MESSAGE = {
    "CLEAR": (
        "🟢 **MALIWANAG (CLEAR)** — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "✅ **STATUS:** Mababa ang posibilidad ng baha sa kasalukuyan.\n"
        "📢 **ACTION:** Walang kinakailangan na aksyon. Patuloy na magbantay.\n\n"
        "📊 **DATA:**\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "Rapid Relay — Flood Early Warning System, Obando"
    ),
    "WATCH": (
        "🟡 **MAGMASID (WATCH)** — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ **STATUS:** May bahagyang banta ng baha sa nakuhang data.\n"
        "📢 **ACTION:** Magmasid sa lebel ng tubig. Maging handa.\n\n"
        "📊 **DATA:**\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "Stay safe. Follow official DRRMO advisories."
    ),
    "WARNING": (
        "🟠 **BABALA (WARNING)** — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ **STATUS:** Mataas ang posibilidad ng pagbaha.\n"
        "📢 **ACTION:** Maghanda na. Itaas ang mga mahahalagang gamit.\n\n"
        "📊 **DATA:**\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "Stay safe. Follow official DRRMO advisories."
    ),
    "DANGER": (
        "🔴 **PELIGRO (DANGER)** — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🚨 **STATUS:** MAY BAHA NA o malapit na ang matinding pagbaha.\n"
        "📢 **ACTION:** LUMIKAS NA AGAD kung ikaw ay nasa mababang lugar.\n\n"
        "📊 **DATA:**\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "‼️ **SUMUNOD SA UTOS NG DRRMO PARA SA EVACUATION.**"
    ),
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_last_posted() -> str | None:
    try:
        if not os.path.exists(STATE_PATH):
            return None
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
            return data.get("last_posted_timestamp")
    except Exception as e:
        print(f"  ⚠️  Could not read state file: {e}")
        return None


def _save_last_posted(timestamp: str) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump({"last_posted_timestamp": timestamp}, f, indent=2)
        print(f"  💾 FB state saved — last posted: {timestamp}")
    except Exception as e:
        print(f"  ⚠️  Could not save state file: {e}")


def _check_credentials() -> bool:
    if not PAGE_ID or not TOKEN:
        print("  ❌ Missing FB credentials.")
        print(f"  📂 .env path: {os.path.abspath(_ENV_PATH)}")
        return False
    return True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_message(tier: str, probability: float, timestamp: str) -> str:
    template = TIER_MESSAGE.get(tier, TIER_MESSAGE["WATCH"])
    return template.format(prob_pct=f"{probability:.1%}", timestamp=timestamp)


def read_latest_from_csv() -> dict | None:
    """Read the latest row from showcase_predict.csv."""
    print(f"  📂 CSV path: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print("  ❌ showcase_predict.csv not found.")
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return None
        required = ["timestamp", "flood_probability", "risk_tier"]
        for col in required:
            if col not in df.columns:
                print(f"  ❌ Missing column: {col}")
                return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        latest = df.sort_values("timestamp").iloc[-1]
        return {
            "tier":        str(latest["risk_tier"]).upper(),
            "probability": float(latest["flood_probability"]),
            "timestamp":   latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception as e:
        print(f"  ❌ Failed to read CSV: {e}")
        return None


def send(tier: str, probability: float, timestamp: str,
         check_duplicate: bool = False) -> bool:
    """
    Post a flood alert to the configured Facebook Page.
    In showcase mode CLEAR is posted — SKIP_TIERS is empty.

    Parameters
    ----------
    tier             : "CLEAR", "WATCH", "WARNING", or "DANGER"
    probability      : float 0.0–1.0
    timestamp        : date string e.g. "2025-07-14"
    check_duplicate  : if True, skips posting if timestamp matches state file

    Returns True on success (or skip), False on any failure.
    """
    if not _check_credentials():
        return False

    # Showcase: no tiers are silently skipped — even CLEAR gets posted
    if tier in SKIP_TIERS:
        print(f"  ⏭️  No post needed — tier is {tier}")
        _save_last_posted(timestamp)
        return True

    if check_duplicate:
        last = _load_last_posted()
        if last and last == timestamp:
            print(f"  ⏭️  [FB] Skipped — already posted for: {timestamp}")
            return True

    message = build_message(tier, probability, timestamp)
    url     = f"https://graph.facebook.com/{VERSION}/{PAGE_ID}/feed"

    emoji = TIER_EMOJI.get(tier, "")
    print(f"  🚀 [FB] Sending {emoji} {tier} alert to Facebook Page")

    try:
        res  = requests.post(url, data={"message": message, "access_token": TOKEN}, timeout=15)
        data = res.json()

        if "id" in data:
            print(f"  ✅ [FB] Posted — post id: {data['id']}")
            _save_last_posted(timestamp)
            return True
        else:
            error = data.get("error", {})
            print(f"  ❌ [FB] API error: {error.get('message', data)}")
            return False

    except requests.Timeout:
        print("  ❌ [FB] Request timed out.")
        return False
    except Exception as e:
        print(f"  ❌ [FB] Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(description="Showcase Facebook alert channel.")
    parser.add_argument("--post", action="store_true",
                        help="Post alert from latest showcase_predict.csv row.")
    parser.add_argument("--test", action="store_true",
                        help="Send a test post with fake data.")
    parser.add_argument("--tier", default="CLEAR",
                        choices=["CLEAR", "WATCH", "WARNING", "DANGER"],
                        help="Tier to use with --test (default: CLEAR).")
    args = parser.parse_args()

    if args.post:
        print(f"\n  🚀 showcase_fb_alert --post")
        row = read_latest_from_csv()
        if row is None:
            print("\n  ❌ FAILED — could not read CSV.")
        else:
            ok = send(tier=row["tier"], probability=row["probability"],
                      timestamp=row["timestamp"], check_duplicate=True)
            print(f"\n  {'✅ SUCCESS' if ok else '❌ FAILED'}")

    elif args.test:
        print(f"\n  🚀 showcase_fb_alert --test — tier={args.tier}")
        ok = send(tier=args.tier, probability=0.12, timestamp=str(date.today()))
        print(f"\n  {'✅ SUCCESS' if ok else '❌ FAILED'}")

    else:
        parser.print_help()

"""
showcase_fb_alert.py
====================
SHOWCASE VERSION of FB_Alert.py

Changes from original:
  - Reads from  : showcase_predict.csv  (instead of flood_rf_sensor_predictions.csv)
  - State file  : showcase_last_fb_posted.json
  - CLEAR tier  : POSTED (not skipped) — showcase dispatches all tiers
  - All Graph API logic mirrors FB_Alert.py exactly.

Usage
-----
    from showcase_fb_alert import send, read_latest_from_csv
    python showcase_fb_alert.py --post
    python showcase_fb_alert.py --test --tier CLEAR
"""

import os
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



def _check_credentials() -> bool:
    if not PAGE_ID or not TOKEN:
        print("  ❌ Missing credentials.")
        print(f"  📂 .env path checked: {os.path.abspath(_ENV_PATH)}")
        print("  ℹ️  Ensure FB_PAGE_ID and FB_PAGE_TOKEN are set in .env")
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
            print("  ❌ CSV is empty.")
            return None

        required = ["timestamp", "flood_probability", "risk_tier"]
        for col in required:
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


def send(tier: str, probability: float, timestamp: str) -> bool:
    """
    Post a flood alert to the configured Facebook Page.
    In showcase mode CLEAR is posted — SKIP_TIERS is empty.
    No duplicate check — always posts on every pipeline run.

    Parameters
    ----------
    tier        : "CLEAR", "WATCH", "WARNING", or "DANGER"
    probability : float 0.0–1.0
    timestamp   : date string e.g. "2025-07-14"

    Returns True on success, False on any failure.
    """
    if not _check_credentials():
        return False

    # Showcase: no tiers are silently skipped — even CLEAR gets posted
    if tier in SKIP_TIERS:
        print(f"  ⏭️  No post needed — tier is {tier} (prob: {probability:.1%}, ts: {timestamp})")
        return True

    message = build_message(tier, probability, timestamp)
    url     = f"https://graph.facebook.com/{VERSION}/{PAGE_ID}/feed"

    emoji = TIER_EMOJI.get(tier, "")
    print(f"  🚀 Sending {emoji} {tier} alert to Facebook Page")
    print(f"  ℹ️  PAGE_ID loaded : {'✅' if PAGE_ID else '❌ MISSING'}")
    print(f"  ℹ️  TOKEN loaded   : {'✅' if TOKEN else '❌ MISSING'}")

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
            print(f"  ✅ [FB] Posted successfully — post id: {data['id']}")
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
        print(f"  ℹ️  PAGE_ID loaded : {'✅' if PAGE_ID else '❌ MISSING'}")
        print(f"  ℹ️  TOKEN loaded   : {'✅' if TOKEN else '❌ MISSING'}")
        print(f"  📂 .env resolved  : {os.path.abspath(_ENV_PATH)}")
        row = read_latest_from_csv()
        if row is None:
            print("\n  ❌ FAILED — could not read CSV.")
        else:
            ok = send(tier=row["tier"], probability=row["probability"],
                      timestamp=row["timestamp"])
            print(f"\n  {'✅ SUCCESS' if ok else '❌ FAILED'}")

    elif args.test:
        print(f"\n  🚀 showcase_fb_alert --test — tier={args.tier}")
        print(f"  ℹ️  PAGE_ID loaded : {'✅' if PAGE_ID else '❌ MISSING'}")
        print(f"  ℹ️  TOKEN loaded   : {'✅' if TOKEN else '❌ MISSING'}")
        print(f"  📂 .env resolved  : {os.path.abspath(_ENV_PATH)}")
        ok = send(tier=args.tier, probability=0.12, timestamp=str(date.today()))
        print(f"\n  {'✅ SUCCESS' if ok else '❌ FAILED'}")

    else:
        parser.print_help()
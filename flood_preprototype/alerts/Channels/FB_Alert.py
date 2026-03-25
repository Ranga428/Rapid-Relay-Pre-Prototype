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
    i.e. D:/Rapid-Relay/Rapid-Relay-Pre-Prototype/.env

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

    # Test manually
    python FB_Alert.py --test
    python FB_Alert.py --test --tier DANGER
"""

import os
import requests
from dotenv import load_dotenv

# .env is two levels above this file (scripts/alerts/ → scripts/ → project root)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

PAGE_ID = os.getenv("FB_PAGE_ID")
TOKEN   = os.getenv("FB_PAGE_TOKEN")

VERSION = "v23.0"

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


def _check_credentials() -> bool:
    """Verify PAGE_ID and TOKEN are loaded. Print helpful error if not."""
    if not PAGE_ID or not TOKEN:
        print("  [FB_Alert] ERROR: Missing credentials.")
        print(f"  [FB_Alert] .env path checked: {os.path.abspath(_ENV_PATH)}")
        print("  [FB_Alert] Ensure FB_PAGE_ID and FB_PAGE_TOKEN are set in .env")
        return False
    return True


def build_message(tier: str, probability: float, timestamp: str) -> str:
    """Build the Facebook post message string for the given tier."""
    template = TIER_MESSAGE.get(tier, TIER_MESSAGE["WARNING"])
    return template.format(
        prob_pct  = f"{probability:.1%}",
        timestamp = timestamp,
    )


def send(tier: str, probability: float, timestamp: str) -> bool:
    """
    Post a flood alert to the configured Facebook Page.

    Parameters
    ----------
    tier        : "WATCH", "WARNING", or "DANGER"
    probability : float 0.0–1.0
    timestamp   : date string e.g. "2025-07-14"

    Returns True on success, False on any failure.
    """
    if not _check_credentials():
        return False

    message = build_message(tier, probability, timestamp)
    url     = f"https://graph.facebook.com/{VERSION}/{PAGE_ID}/feed"

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
            print(f"  [FB_Alert] ✓ Posted — post id: {data['id']}")
            return True
        else:
            error = data.get("error", {})
            print(f"  [FB_Alert] ✗ API error: {error.get('message', data)}")
            return False

    except requests.Timeout:
        print("  [FB_Alert] ✗ Request timed out.")
        return False
    except Exception as e:
        print(f"  [FB_Alert] ✗ Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Facebook alert channel.")
    parser.add_argument("--test", action="store_true", help="Send a test post.")
    parser.add_argument(
        "--tier", default="WARNING",
        choices=["WATCH", "WARNING", "DANGER"],
        help="Tier to test (default: WARNING).",
    )
    args = parser.parse_args()

    if args.test:
        from datetime import date
        print(f"\n  FB_Alert test — tier={args.tier}")
        print(f"  PAGE_ID loaded : {'✓' if PAGE_ID else '✗ MISSING'}")
        print(f"  TOKEN loaded   : {'✓' if TOKEN else '✗ MISSING'}")
        ok = send(tier=args.tier, probability=0.55, timestamp=str(date.today()))
        print(f"\n  Result: {'SUCCESS' if ok else 'FAILED'}")
    else:
        parser.print_help()
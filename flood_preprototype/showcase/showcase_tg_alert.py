"""
showcase_tg_alert.py
====================
SHOWCASE VERSION of TG_Alert.py

Changes from original:
  - Reads from  : showcase_predict.csv  (instead of flood_rf_sensor_predictions.csv)
  - State file  : showcase_last_tg_sent.json
  - CLEAR tier  : SENT (not skipped) — showcase dispatches all tiers
  - All Telegram Bot API logic is identical to the original.

Usage
-----
    from showcase_tg_alert import send
    python showcase_tg_alert.py
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH  = os.path.join(SCRIPT_DIR, "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# Reads showcase_predict.csv
PREDICTIONS_CSV = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "predictions", "showcase_predict.csv")
)

STATE_PATH = os.path.join(SCRIPT_DIR, "showcase_last_tg_sent.json")

# Showcase: CLEAR tier is sent (not skipped)
SKIP_TIERS: set = set()

TIER_MESSAGE = {
    "CLEAR": (
        "🟢 *MALIWANAG (CLEAR)* — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "✅ *STATUS:* Mababa ang posibilidad ng baha sa kasalukuyan.\n"
        "📢 *ACTION:* Walang kinakailangan na aksyon. Patuloy na magbantay.\n\n"
        "📊 *DATA:*\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "Rapid Relay — Flood Early Warning System, Obando"
    ),
    "WATCH": (
        "🟡 *MAGMASID (WATCH)* — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *STATUS:* May bahagyang banta ng baha sa nakuhang data.\n"
        "📢 *ACTION:* Magmasid sa lebel ng tubig. Maging handa.\n\n"
        "📊 *DATA:*\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "Stay safe. Follow official DRRMO advisories."
    ),
    "WARNING": (
        "🟠 *BABALA (WARNING)* — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *STATUS:* Mataas ang posibilidad ng pagbaha.\n"
        "📢 *ACTION:* Maghanda na. Itaas ang mga mahahalagang gamit.\n\n"
        "📊 *DATA:*\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "Stay safe. Follow official DRRMO advisories."
    ),
    "DANGER": (
        "🔴 *PELIGRO (DANGER)* — Obando, Bulacan\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🚨 *STATUS:* MAY BAHA NA o malapit na ang matinding pagbaha.\n"
        "📢 *ACTION:* LUMIKAS NA AGAD kung ikaw ay nasa mababang lugar.\n\n"
        "📊 *DATA:*\n"
        "• Prob. ng Baha: {prob_pct}\n"
        "• Oras: {timestamp}\n\n"
        "‼️ *SUMUNOD SA UTOS NG DRRMO PARA SA EVACUATION.*"
    ),
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_last_sent() -> str | None:
    try:
        if not os.path.exists(STATE_PATH):
            return None
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
            return data.get("last_telegram_timestamp")
    except Exception as e:
        print(f"  ⚠️  Could not read TG state file: {e}")
        return None


def _save_last_sent(timestamp: str) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump({"last_telegram_timestamp": timestamp}, f, indent=2)
        print(f"  💾 TG state saved — last sent: {timestamp}")
    except Exception as e:
        print(f"  ⚠️  Could not save TG state file: {e}")


# ---------------------------------------------------------------------------
# Telegram core
# ---------------------------------------------------------------------------

def read_latest_prediction() -> dict | None:
    if not os.path.exists(PREDICTIONS_CSV):
        print("  ❌ showcase_predict.csv not found.")
        return None
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        latest = df.sort_values("timestamp").iloc[-1]
        return {
            "tier":        str(latest["risk_tier"]).upper(),
            "probability": float(latest["flood_probability"]),
            "timestamp":   latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception as e:
        print(f"  ❌ Failed to read showcase_predict.csv: {e}")
        return None


def send_telegram_message(token: str, chat_id: str, message_content: str) -> bool:
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       message_content,
        "parse_mode": "Markdown",
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("  ✅ [TG] Message sent successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ❌ [TG] Failed to send message: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  🔍 Telegram API Error: {e.response.text}")
        return False


# ---------------------------------------------------------------------------
# Standard channel interface
# ---------------------------------------------------------------------------

def send(tier: str, probability: float, timestamp: str,
         check_duplicate: bool = False) -> bool:
    """
    Standard send() interface for showcase_alert.py integration.

    Showcase mode: CLEAR tier is also sent. SKIP_TIERS is empty.

    Parameters
    ----------
    tier             : "CLEAR", "WATCH", "WARNING", or "DANGER"
    probability      : float 0.0–1.0
    timestamp        : date string
    check_duplicate  : if True, skips if this timestamp was already sent

    Returns True if sent successfully, False if skipped or failed.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [TG_Alert] Missing credentials — skipping.")
        return False

    # Showcase: all tiers are sent
    if tier in SKIP_TIERS:
        print(f"  [TG_Alert] Tier is {tier} — skipping (in SKIP_TIERS).")
        return False

    template = TIER_MESSAGE.get(tier)
    if not template:
        print(f"  [TG_Alert] Unknown tier '{tier}' — skipping.")
        return False

    final_message = template.format(
        prob_pct  = f"{probability:.1%}",
        timestamp = timestamp,
    )

    success = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_message)

    if success:
        _save_last_sent(timestamp)

    return success


# ---------------------------------------------------------------------------
# CLI / standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🚀 showcase_tg_alert — Telegram Alert (Showcase Mode)")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ❌ Missing Telegram credentials in .env")
        exit(1)

    prediction = read_latest_prediction()
    if not prediction:
        exit(1)

    tier = prediction["tier"]
    prob = prediction["probability"]
    ts   = prediction["timestamp"]
    print(f"  ℹ️  Latest — Tier: {tier} | Prob: {prob:.1%} | Time: {ts}")

    if tier in SKIP_TIERS:
        print(f"  ⏭️  Tier {tier} is in SKIP_TIERS — no message sent.")
        _save_last_sent(ts)
        exit(0)
        
    template = TIER_MESSAGE.get(tier)
    if not template:
        print(f"  ❌ Unknown tier '{tier}'.")
        exit(1)

    final_message = template.format(prob_pct=f"{prob:.1%}", timestamp=ts)

    print("\n--- Message Preview ---")
    print(final_message)
    print("-----------------------\n")

    success = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_message)

    if success:
        _save_last_sent(ts)
        print("  🎉 Telegram alert sent successfully.")
    else:
        print("  ⚠️  Alert failed. State NOT saved (allows retry).")

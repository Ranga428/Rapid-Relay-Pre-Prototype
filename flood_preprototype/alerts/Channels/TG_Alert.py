# TG_Alert.py

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# --- Paths & Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

# Fetch Telegram Credentials securely
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Default CSV path — can be overridden via --csv flag or csv_path argument
DEFAULT_PREDICTIONS_CSV = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "predictions", "flood_rf_sensor_predictions.csv")
)

STATE_PATH = os.path.join(SCRIPT_DIR, "last_telegram_sent.json")

# --- Constants ---
SKIP_TIERS = {"CLEAR", "LOW", "NORMAL"}

TIER_MESSAGE = {
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

# --- State Management ---
def _load_last_sent() -> str | None:
    try:
        if not os.path.exists(STATE_PATH):
            return None
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
            return data.get("last_telegram_timestamp")
    except Exception as e:
        print(f"  ⚠️  Could not read state file: {e}")
        return None

def _save_last_sent(timestamp: str) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump({"last_telegram_timestamp": timestamp}, f, indent=2)
        print(f"  💾 State saved — last Telegram sent for: {timestamp}")
    except Exception as e:
        print(f"  ⚠️  Could not save state file: {e}")

# --- Data Readers ---
def read_latest_prediction(csv_path: str = None) -> dict | None:
    """
    Read the last row (by timestamp) from the predictions CSV.

    Parameters
    ----------
    csv_path : optional path to a CSV file. Defaults to DEFAULT_PREDICTIONS_CSV.
    """
    path = csv_path or DEFAULT_PREDICTIONS_CSV
    if not os.path.exists(path):
        print(f"  ❌ Predictions CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print("  ❌ Predictions CSV is empty.")
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        latest = df.sort_values("timestamp").iloc[-1]
        return {
            "tier":        str(latest["risk_tier"]).upper(),
            "probability": float(latest["flood_probability"]),
            "timestamp":   latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception as e:
        print(f"  ❌ Failed to read predictions CSV: {e}")
        return None

# --- Telegram Logic ---
def send_telegram_alert(token, chat_id, message_content) -> bool:
    """Sends a message to a Telegram Group Chat."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_content,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("  ✅ Message successfully posted to Telegram Group!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to send Telegram message: {e}")
        if e.response is not None:
            print(f"  🔍 Telegram API Error: {e.response.text}")
        return False

# --- Standard Channel Interface ---
def send(tier: str, probability: float, timestamp: str, check_duplicate: bool = True) -> bool:
    """
    Standard send() interface for Alert.py integration.

    Parameters
    ----------
    tier            : "CLEAR", "WATCH", "WARNING", or "DANGER"
    probability     : float 0.0–1.0
    timestamp       : date string e.g. "2025-07-14"
    check_duplicate : if True, skips if this timestamp was already sent
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [TG_Alert] Missing credentials — skipping.")
        return False

    if tier in SKIP_TIERS:
        print(f"  [TG_Alert] Tier is {tier} — skipping.")
        return False

    if check_duplicate:
        last_sent = _load_last_sent()
        if last_sent == timestamp:
            print(f"  [TG_Alert] Already sent for {timestamp} — skipping.")
            return False

    template = TIER_MESSAGE.get(tier)
    if not template:
        print(f"  [TG_Alert] Unknown tier '{tier}' — skipping.")
        return False

    final_message = template.format(prob_pct=f"{probability:.1%}", timestamp=timestamp)
    success = send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_message)

    if success:
        _save_last_sent(timestamp)

    return success


# --- Main Execution ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Telegram alert channel.")
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Override the CSV file to read from (default: flood_rf_sensor_predictions.csv).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Send the alert even if it was already sent for this timestamp.",
    )
    args = parser.parse_args()

    print("\n🚀 Starting Telegram Alert Routine...")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ❌ Missing Telegram credentials in .env file.")
        exit(1)

    prediction = read_latest_prediction(csv_path=args.csv)
    if not prediction:
        exit(1)

    tier = prediction["tier"]
    prob = prediction["probability"]
    ts   = prediction["timestamp"]

    print(f"  ℹ️  Latest Data — Tier: {tier} | Prob: {prob:.1%} | Time: {ts}")

    if tier in SKIP_TIERS:
        print(f"  ⏭️  No alert needed — tier is {tier}.")
        _save_last_sent(ts)
        exit(0)

    last_sent = _load_last_sent()
    if last_sent == ts and not args.force:  # Added "and not args.force"
        print(f"  ⏭️  Skipped — Telegram alert already sent for timestamp: {ts}")
        exit(0)

    template = TIER_MESSAGE.get(tier)
    if not template:
        print(f"  ❌ Aborting: Unknown tier '{tier}'.")
        exit(1)

    final_message = template.format(prob_pct=f"{prob:.1%}", timestamp=ts)

    print("\n--- Message Preview ---")
    print(final_message)
    print("-----------------------\n")

    success = send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, final_message)

    if success:
        _save_last_sent(ts)
        print("  🎉 Telegram Alert Routine completed successfully.")
    else:
        print("  ⚠️  Routine finished with errors. State not saved to allow retry.")
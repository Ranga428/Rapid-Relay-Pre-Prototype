"""
SMS_Alert.py
============
Flood EWS — SMS Alert Channel (stub)

Sends flood alerts via SMS using Twilio (or any compatible gateway).

CREDENTIALS (.env — two levels above this file)
------------------------------------------------
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TWILIO_FROM=+1xxxxxxxxxx        ← your Twilio number
    ALERT_SMS_TO=+63xxxxxxxxxx      ← recipient number (or comma-separated list)

SETUP
-----
    pip install twilio
    Add credentials to .env
    Uncomment the implementation in send() below

Usage
-----
    from alerts.SMS_Alert import send
    ok = send(tier="WARNING", probability=0.52, timestamp="2025-07-14")

    python SMS_Alert.py --test
"""

import os
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM")
TO_NUMBERS  = [
    n.strip()
    for n in (os.getenv("ALERT_SMS_TO") or "").split(",")
    if n.strip()
]

TIER_EMOJI = {"WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴"}


def send(tier: str, probability: float, timestamp: str) -> bool:
    """
    Send an SMS flood alert.
    Returns True on success, False if not configured or on failure.
    """
    if not all([ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBERS]):
        print("  [SMS_Alert] Not configured — skipping.")
        return False

    emoji   = TIER_EMOJI.get(tier, "")
    message = (
        f"{emoji} RAPID RELAY {tier} — Obando, Bulacan\n"
        f"Flood probability: {probability:.1%} on {timestamp}.\n"
        f"Follow DRRMO advisories."
    )

    # ── Uncomment when Twilio is configured ──────────────────────────────
    # from twilio.rest import Client
    # client  = Client(ACCOUNT_SID, AUTH_TOKEN)
    # results = []
    # for number in TO_NUMBERS:
    #     try:
    #         msg = client.messages.create(body=message, from_=FROM_NUMBER, to=number)
    #         print(f"  [SMS_Alert] ✓ Sent to {number} — sid: {msg.sid}")
    #         results.append(True)
    #     except Exception as e:
    #         print(f"  [SMS_Alert] ✗ Failed to {number}: {e}")
    #         results.append(False)
    # return any(results)
    # ─────────────────────────────────────────────────────────────────────

    print("  [SMS_Alert] Not yet implemented — uncomment Twilio block in SMS_Alert.py")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test SMS alert channel.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tier", default="WARNING", choices=["WATCH", "WARNING", "DANGER"])
    args = parser.parse_args()
    if args.test:
        from datetime import date
        print(f"  SMS_Alert test — tier={args.tier}")
        ok = send(tier=args.tier, probability=0.55, timestamp=str(date.today()))
        print(f"  Result: {'SUCCESS' if ok else 'FAILED/NOT CONFIGURED'}")
    else:
        parser.print_help()
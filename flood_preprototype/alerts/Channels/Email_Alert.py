"""
Email_Alert.py
==============
Flood EWS — Email Alert Channel (stub)

Sends flood alerts via SMTP.

CREDENTIALS (.env — two levels above this file)
------------------------------------------------
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your@gmail.com
    SMTP_PASS=your_app_password       ← use App Password, not account password
    ALERT_EMAIL_TO=recipient@email.com  ← comma-separated for multiple

SETUP
-----
    No extra pip installs needed (uses stdlib smtplib).
    Add credentials to .env
    Uncomment the implementation in send() below.

Usage
-----
    from alerts.Email_Alert import send
    ok = send(tier="WARNING", probability=0.52, timestamp="2025-07-14")

    python Email_Alert.py --test
"""

import os
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

SMTP_HOST  = os.getenv("SMTP_HOST")
SMTP_PORT  = int(os.getenv("SMTP_PORT", 587))
SMTP_USER  = os.getenv("SMTP_USER")
SMTP_PASS  = os.getenv("SMTP_PASS")
TO_EMAILS  = [
    e.strip()
    for e in (os.getenv("ALERT_EMAIL_TO") or "").split(",")
    if e.strip()
]

TIER_EMOJI = {"WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴"}


def send(tier: str, probability: float, timestamp: str) -> bool:
    """
    Send an email flood alert.
    Returns True on success, False if not configured or on failure.
    """
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, TO_EMAILS]):
        print("  [Email_Alert] Not configured — skipping.")
        return False

    emoji   = TIER_EMOJI.get(tier, "")
    subject = f"{emoji} Rapid Relay {tier} Alert — Obando, Bulacan"
    body    = (
        f"Rapid Relay Flood Early Warning System\n"
        f"{'=' * 45}\n\n"
        f"Alert Tier   : {tier}\n"
        f"Probability  : {probability:.1%}\n"
        f"Date         : {timestamp}\n"
        f"Location     : Obando, Bulacan, Philippines\n\n"
        f"Please follow advisories from your local DRRMO.\n"
    )

    # ── Uncomment when SMTP is configured ────────────────────────────────
    # import smtplib
    # from email.mime.text import MIMEText
    # from email.mime.multipart import MIMEMultipart
    #
    # msg = MIMEMultipart()
    # msg["From"]    = SMTP_USER
    # msg["To"]      = ", ".join(TO_EMAILS)
    # msg["Subject"] = subject
    # msg.attach(MIMEText(body, "plain"))
    #
    # try:
    #     with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
    #         server.starttls()
    #         server.login(SMTP_USER, SMTP_PASS)
    #         server.sendmail(SMTP_USER, TO_EMAILS, msg.as_string())
    #     print(f"  [Email_Alert] ✓ Sent to {', '.join(TO_EMAILS)}")
    #     return True
    # except Exception as e:
    #     print(f"  [Email_Alert] ✗ Failed: {e}")
    #     return False
    # ─────────────────────────────────────────────────────────────────────

    print("  [Email_Alert] Not yet implemented — uncomment SMTP block in Email_Alert.py")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test email alert channel.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tier", default="WARNING", choices=["WATCH", "WARNING", "DANGER"])
    args = parser.parse_args()
    if args.test:
        from datetime import date
        print(f"  Email_Alert test — tier={args.tier}")
        ok = send(tier=args.tier, probability=0.55, timestamp=str(date.today()))
        print(f"  Result: {'SUCCESS' if ok else 'FAILED/NOT CONFIGURED'}")
    else:
        parser.print_help()
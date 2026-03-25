"""
Alert.py
========
Flood EWS — Alert Coordinator

Calls all configured alert channels when a flood tier is detected.
This is the ONLY file Start.py needs to call.

CHANNEL MODULES (in alerts/)
-----------------------------
    alerts/FB_Alert.py       — Facebook Page post (Graph API)
    alerts/SMS_Alert.py      — SMS via Twilio          [stub — configure to activate]
    alerts/Email_Alert.py    — Email via SMTP           [stub — configure to activate]
    alerts/Push_Alert.py     — App push via FCM         [stub — configure to activate]
    alerts/Webhook_Alert.py  — Generic HTTP webhook     [stub — configure to activate]

Each channel module:
    - Loads its own credentials from .env (two levels above)
    - Exposes a single send(tier, probability, timestamp) -> bool
    - Returns False silently if not configured (no crash)
    - Can be tested independently: python alerts/FB_Alert.py --test

ADDING A NEW CHANNEL
--------------------
    1. Create alerts/MyChannel_Alert.py with a send() function
    2. Import it here and add it to CHANNELS below
    3. Start.py needs no changes

TIERS THAT TRIGGER ALERTS
--------------------------
    WATCH   — elevated risk
    WARNING — high risk
    DANGER  — imminent / ongoing flood
    CLEAR   — no alert sent

Usage
-----
    # Called by Start.py
    import Alert
    Alert.dispatch_alert(tier="WARNING", probability=0.52, timestamp="2025-07-14")

    # Test all channels
    python Alert.py --test --tier WARNING

    # Test a single channel directly
    python alerts/FB_Alert.py --test --tier WARNING
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from Channels import FB_Alert, SMS_Alert, Email_Alert, Push_Alert, Webhook_Alert

# Tiers that trigger outbound alerts
ALERT_TIERS = {"WATCH", "WARNING", "DANGER"}

TIER_EMOJI = {
    "CLEAR":   "🟢",
    "WATCH":   "🟡",
    "WARNING": "🟠",
    "DANGER":  "🔴",
}

# Channel registry — add new channels here
# Format: (label, module)
CHANNELS = [
    ("facebook", FB_Alert),
    ("sms",      SMS_Alert),
    ("email",    Email_Alert),
    ("push",     Push_Alert),
    ("webhook",  Webhook_Alert),
]


def dispatch_alert(tier: str, probability: float, timestamp: str) -> dict:
    """
    Fire all configured alert channels for the given prediction.

    Parameters
    ----------
    tier        : "CLEAR", "WATCH", "WARNING", or "DANGER"
    probability : float 0.0-1.0
    timestamp   : date string e.g. "2025-07-14"

    Returns
    -------
    dict of {channel_label: bool} — True if channel fired successfully.
    All values are False for CLEAR (no alert sent).
    """
    results = {label: False for label, _ in CHANNELS}

    if tier not in ALERT_TIERS:
        return results

    emoji = TIER_EMOJI.get(tier, "")
    print(f"\n  [Alert] Dispatching {emoji} {tier} — {probability:.1%} on {timestamp}")

    for label, module in CHANNELS:
        try:
            results[label] = module.send(
                tier=tier,
                probability=probability,
                timestamp=timestamp,
            )
        except Exception as e:
            print(f"  [Alert] {label} raised an exception: {e}")
            results[label] = False

    # Summary
    fired   = [ch for ch, ok in results.items() if ok]
    skipped = [ch for ch, ok in results.items() if not ok]
    if fired:
        print(f"  [Alert] Fired    : {', '.join(fired)}")
    if skipped:
        print(f"  [Alert] Skipped  : {', '.join(skipped)}")

    return results


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(description="Test all alert channels.")
    parser.add_argument("--test",  action="store_true", help="Fire a test alert.")
    parser.add_argument(
        "--tier", default="WARNING",
        choices=["WATCH", "WARNING", "DANGER"],
        help="Alert tier to test (default: WARNING).",
    )
    args = parser.parse_args()

    if args.test:
        print(f"\n  Alert coordinator test — tier={args.tier}")
        results = dispatch_alert(
            tier=args.tier,
            probability=0.55,
            timestamp=str(date.today()),
        )
        print(f"\n  Channel results:")
        for ch, ok in results.items():
            status = "✓ OK" if ok else "✗ skipped/failed"
            print(f"    {ch:<12} {status}")
    else:
        parser.print_help()
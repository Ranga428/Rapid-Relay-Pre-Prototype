"""
Alert.py
========
Flood EWS — Alert Coordinator

Calls all configured alert channels when a flood tier is detected.
This is the ONLY file Start.py needs to call.

CHANNEL MODULES (in alerts/)
-----------------------------
    alerts/FB_Alert.py          — Facebook Page post (Graph API)
    alerts/Supabase_Alert.py    — Supabase sync + alert record
    alerts/SMS_Alert.py         — SMS via Twilio          [stub — configure to activate]
    alerts/Email_Alert.py       — Email via SMTP           [stub — configure to activate]
    alerts/Push_Alert.py        — App push via FCM         [stub — configure to activate]
    alerts/Webhook_Alert.py     — Generic HTTP webhook     [stub — configure to activate]

Each channel module:
    - Loads its own credentials from .env (project root)
    - Exposes a single send(tier, probability, timestamp) -> bool
      (Supabase_Alert wraps its sync + alert functions behind this interface)
    - Returns False silently if not configured (no crash)
    - Can be tested independently: python alerts/FB_Alert.py --test

CHANNEL BEHAVIOUR
-----------------
    ALWAYS_FIRE_CHANNELS  — fire on every call, including CLEAR.
                            Facebook and Supabase are always notified so the
                            dashboard and page stay in sync regardless of tier.
    CHANNELS              — fire only when tier is WATCH / WARNING / DANGER.

    Facebook specifically:
        - SKIP_TIERS (CLEAR, LOW, NORMAL) → terminal notice only, no API call
        - check_duplicate=True            → skips if timestamp already posted

ADDING A NEW CHANNEL
--------------------
    1. Create alerts/MyChannel_Alert.py with a send() function
    2. Import it and add it to ALWAYS_FIRE_CHANNELS or CHANNELS below
    3. Start.py needs no changes

TIERS THAT TRIGGER ALERTS
--------------------------
    WATCH   — elevated risk
    WARNING — high risk
    DANGER  — imminent / ongoing flood
    CLEAR   — FB + Supabase still fire; all other channels skipped

Usage
-----
    # Called by Start.py
    import Alert
    Alert.dispatch_alert(tier="WARNING", probability=0.52, timestamp="2025-07-14")

    # Run directly — reads latest CSV row and dispatches all channels
    python Alert.py

    # Test a single channel directly
    python alerts/FB_Alert.py --test --tier WARNING
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from Channels import FB_Alert
import Channels.Supabase_Alert as _supabase_module
from Channels import TG_Alert   


# ---------------------------------------------------------------------------
# Supabase channel shim
# Supabase_Alert exposes sync/append functions rather than send().
# This thin wrapper plugs it into the standard channel interface without
# modifying Supabase_Alert.py itself.
# ---------------------------------------------------------------------------
class _SupabaseChannel:
    """Wraps Supabase_Alert behind the standard send() interface."""

    @staticmethod
    def send(tier: str, probability: float, timestamp: str) -> bool:
        try:
            # Append new prediction rows from the latest CSV
            _supabase_module.append_new_predictions_to_supabase()
            return True
        except Exception as e:
            print(f"  [Supabase_Alert] send() shim raised: {e}")
            return False


# Tiers that trigger the gated channels (SMS, Email, Push, Webhook)
ALERT_TIERS = {"WATCH", "WARNING", "DANGER"}

TIER_EMOJI = {
    "CLEAR":   "🟢",
    "WATCH":   "🟡",
    "WARNING": "🟠",
    "DANGER":  "🔴",
}

# Always fire — every tier including CLEAR
ALWAYS_FIRE_CHANNELS = [
    ("facebook", FB_Alert),
    ("supabase", _SupabaseChannel),
]

# Tier-gated — only fire on WATCH / WARNING / DANGER
CHANNELS = [
    ("telegram", TG_Alert),
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

    Channel behaviour
    -----------------
    ALWAYS_FIRE_CHANNELS  — fire on every call, including CLEAR.
                            Facebook and Supabase are always notified so the
                            dashboard and page stay in sync regardless of tier.
    CHANNELS              — fire only when tier is WATCH / WARNING / DANGER.
    """
    results = {label: False for label, _ in ALWAYS_FIRE_CHANNELS + CHANNELS}

    emoji = TIER_EMOJI.get(tier, "")
    print(f"\n  [Alert] Dispatching {emoji} {tier} — {probability:.1%} on {timestamp}")

    # --- Always-fire channels (FB + Supabase) ---
    for label, module in ALWAYS_FIRE_CHANNELS:
        try:
            # FB_Alert: enable SKIP_TIERS + anti-duplicate check
            if label == "facebook":
                results[label] = module.send(
                    tier=tier,
                    probability=probability,
                    timestamp=timestamp,
                    check_duplicate=True,
                )
            else:
                results[label] = module.send(
                    tier=tier,
                    probability=probability,
                    timestamp=timestamp,
                )
        except Exception as e:
            print(f"  [Alert] {label} raised an exception: {e}")
            results[label] = False

    # --- Tier-gated channels (SMS, Email, Push, Webhook) ---
    if tier in ALERT_TIERS:
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
# CLI — reads latest CSV row and dispatches to all channels
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from Channels.FB_Alert import read_latest_from_csv

    print("\n  🚀 Alert.py — reading latest CSV row")
    row = read_latest_from_csv()

    if row is None:
        print("  ❌ Could not read CSV. Aborting.")
    else:
        results = dispatch_alert(
            tier=row["tier"],
            probability=row["probability"],
            timestamp=row["timestamp"],
        )
        print(f"\n  Channel results:")
        for ch, ok in results.items():
            status = "✅ OK" if ok else "❌ failed"
            print(f"    {ch:<12} {status}")
"""
showcase_alert.py
=================
SHOWCASE VERSION of Alert.py

Changes from original:
  - Reads from showcase_predict.csv  (instead of flood_xgb_sensor_predictions.csv)
  - Dispatches ALL tiers including CLEAR — no tier is silently skipped.
  - FB_Alert.py is called with SKIP_TIERS overridden so CLEAR posts are sent.
  - TG_Alert.py is called for ALL tiers (CLEAR, WATCH, WARNING, DANGER).
  - Supabase_Alert syncs showcase_predict.csv (via the same append logic).
  - All channel modules (FB_Alert, TG_Alert, Supabase_Alert) are imported
    from the showcase/ folder since they are cloned there.
  - All other coordinator logic is identical to the original.

Usage
-----
    # Called by showcase_start.py
    import showcase_alert
    showcase_alert.dispatch_alert(tier="CLEAR", probability=0.12, timestamp="2025-07-14")

    # Run directly — reads latest showcase_predict.csv row
    python showcase_alert.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import showcase versions of each channel module
from showcase_fb_alert       import send as _fb_send
from showcase_tg_alert       import send as _tg_send
import showcase_supabase_alert as _supabase_module


# ---------------------------------------------------------------------------
# Supabase channel shim  (identical pattern to original Alert.py)
# ---------------------------------------------------------------------------

class _SupabaseChannel:
    @staticmethod
    def send(tier: str, probability: float, timestamp: str) -> bool:
        try:
            _supabase_module.append_new_predictions_to_supabase()
            return True
        except Exception as e:
            print(f"  [Supabase_Alert] send() shim raised: {e}")
            return False


# ---------------------------------------------------------------------------
# Facebook channel shim — wraps showcase_fb_alert.send()
# ---------------------------------------------------------------------------

class _FBChannel:
    @staticmethod
    def send(tier: str, probability: float, timestamp: str,
             check_duplicate: bool = True) -> bool:
        return _fb_send(
            tier=tier,
            probability=probability,
            timestamp=timestamp,
            check_duplicate=check_duplicate,
        )


# ---------------------------------------------------------------------------
# Telegram channel shim — wraps showcase_tg_alert.send()
# ---------------------------------------------------------------------------

class _TGChannel:
    @staticmethod
    def send(tier: str, probability: float, timestamp: str) -> bool:
        return _tg_send(
            tier=tier,
            probability=probability,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# NOTE: In the showcase all four tiers trigger all channels.
# There is NO silent skip for CLEAR — every prediction is dispatched.
ALL_TIERS = {"CLEAR", "WATCH", "WARNING", "DANGER"}

TIER_EMOJI = {
    "CLEAR":   "🟢",
    "WATCH":   "🟡",
    "WARNING": "🟠",
    "DANGER":  "🔴",
}

# Channels that fire on every tier (including CLEAR)
ALWAYS_FIRE_CHANNELS = [
    ("facebook", _FBChannel),
    ("supabase", _SupabaseChannel),
    ("telegram", _TGChannel),
]

# No tier-gated channels in the showcase — all tiers go to all channels.
CHANNELS = []


def dispatch_alert(tier: str, probability: float, timestamp: str) -> dict:
    """
    Fire all configured alert channels for the given prediction.

    Unlike the original Alert.py, the showcase dispatches ALL tiers —
    including CLEAR — to all channels. This lets the showcase demonstrate
    the full transmission pipeline for every prediction.

    Parameters
    ----------
    tier        : "CLEAR", "WATCH", "WARNING", or "DANGER"
    probability : float 0.0–1.0
    timestamp   : date string e.g. "2025-07-14"

    Returns
    -------
    dict of {channel_label: bool}
    """
    results = {label: False for label, _ in ALWAYS_FIRE_CHANNELS + CHANNELS}
    emoji   = TIER_EMOJI.get(tier, "")
    print(f"\n  [showcase_alert] Dispatching {emoji} {tier} — {probability:.1%} on {timestamp}")
    print(f"  [showcase_alert] All tiers are dispatched in showcase mode (including CLEAR)")

    # All channels fire for all tiers
    for label, module in ALWAYS_FIRE_CHANNELS:
        try:
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
            print(f"  [showcase_alert] {label} raised an exception: {e}")
            results[label] = False

    fired   = [ch for ch, ok in results.items() if ok]
    skipped = [ch for ch, ok in results.items() if not ok]
    if fired:
        print(f"  [showcase_alert] Fired    : {', '.join(fired)}")
    if skipped:
        print(f"  [showcase_alert] Skipped  : {', '.join(skipped)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from showcase_fb_alert import read_latest_from_csv

    print("\n  🚀 showcase_alert.py — reading latest showcase_predict.csv row")
    row = read_latest_from_csv()

    if row is None:
        print("  ❌ Could not read CSV. Aborting.")
    else:
        results = dispatch_alert(
            tier        = row["tier"],
            probability = row["probability"],
            timestamp   = row["timestamp"],
        )
        print(f"\n  Channel results:")
        for ch, ok in results.items():
            status = "✅ OK" if ok else "❌ failed"
            print(f"    {ch:<12} {status}")

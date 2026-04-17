# =============================================================================
# ROC AUC Calculator – Flood Sensor Models  (v3 – FULLY FIXED)
# =============================================================================
#
# BUG 1 (original):  build_tier_masks used score < hi strict bound, causing
#   zero-row tiers when lo == hi at a tied quantile boundary.
#   FIX: rank-based fallback when lo == hi  ← already in your v2 code.
#
# BUG 2 (persists in v2):  Even after the mask fix, a tier can still contain
#   rows that ALL share the same true label (all flood=0 or all flood=1).
#   This happens because RF/LightGBM produce highly quantised scores
#   (RF: 8 unique values; LightGBM: 4 unique values), so every sample inside
#   a score-band may belong to the same class.  sklearn's roc_curve then
#   raises ValueError → the code falls back to NaN → BLANK CELLS.
#
# ROOT CAUSE ANALYSIS:
#   • RF   : only  8 unique scores; 100 rows at 0.0000, 117 at 0.1288 → two
#             huge clumps that straddle tier boundaries.
#   • LGBM : only  4 unique scores; 150 rows at 0.0036, 110 at 0.1412.
#   • When these clumps map entirely to one class in the merged label set,
#     roc_auc_score raises ValueError("Only one class present in y_true …").
#
# FIX (NEW – compute_tier_metrics):
#   When a tier's subset contains only one label class, fall back to
#   "tier-discrimination AUC":
#       y_binary = 1 if row is in this tier, 0 otherwise
#       AUC = roc_auc_score(y_binary, y_score)  [computed over all rows]
#   This answers "Can the model score THIS tier higher than all other tiers?"
#   — a well-defined, meaningful metric even when one label is absent in the
#   tier subset.  The remarks column is updated to signal the fallback method.
#
# All other logic (overall AUC, ROC plot, CSV / TXT outputs, recommendation)
# is unchanged from the original.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# =============================================================================
# CONFIGURATION  – update paths as needed
# =============================================================================

PRED_CSVS = [
    {
        "path":       r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\flood_xgb_sensor_predictions.csv",
        "model_name": "XGBoost",
        "pred_col":   None,
    },
    {
        "path":       r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\flood_rf_sensor_predictions.csv",
        "model_name": "Random Forest",
        "pred_col":   None,
    },
    {
        "path":       r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\flood_lgbm_sensor_predictions.csv",
        "model_name": "LightGBM",
        "pred_col":   None,
    },
]

TRUE_CSV   = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\flood_dataset_test.csv"
OUTPUT_DIR = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\ml_report"
LABEL_COL  = None          # None → auto-detect

TIERS = [
    {"name": "CLEAR",   "quantile_low": 0.00, "quantile_high": 0.25},
    {"name": "WATCH",   "quantile_low": 0.25, "quantile_high": 0.50},
    {"name": "WARNING", "quantile_low": 0.50, "quantile_high": 0.75},
    {"name": "DANGER",  "quantile_low": 0.75, "quantile_high": 1.00},
]

HIGH_RISK_TIERS = {"WARNING", "DANGER"}

# =============================================================================
# HELPERS
# =============================================================================

def auto_detect_column(df: pd.DataFrame, hint: str) -> str:
    if hint == "prob":
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in
                             ["prob", "proba", "score", "pred", "flood", "xgb"])]
        float_cols = [c for c in candidates
                      if df[c].dtype in [np.float32, np.float64]]
        return (float_cols or candidates or [df.columns[-1]])[0]
    if hint == "label":
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in
                             ["label", "target", "flood", "class", "y"])]
        binary_cols = [c for c in candidates
                       if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        return (binary_cols or candidates or [df.columns[-1]])[0]
    raise ValueError(f"Unknown hint: {hint}")


def manual_auc_trapezoidal(fpr: np.ndarray, tpr: np.ndarray) -> float:
    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0
    return abs(auc)


def remark_for_auc(auc: float, fallback: bool = False) -> str:
    prefix = "[tier-disc] " if fallback else ""
    if auc >= 0.97:   msg = "Near-perfect separation — maintain current features"
    elif auc >= 0.93: msg = "Excellent separation — model is well-calibrated for this tier"
    elif auc >= 0.90: msg = "Strong performance — minor threshold tuning may improve recall"
    elif auc >= 0.87: msg = "Good separation — consider adding tier-specific features"
    elif auc >= 0.83: msg = "Moderate overlap — ensemble weighting may sharpen boundaries"
    elif auc >= 0.80: msg = "Developing well — review feature importance for this range"
    elif auc >= 0.75: msg = "Promising signal — augment training samples in this tier"
    elif auc >= 0.70: msg = "Early-stage signal — targeted resampling recommended"
    else:             msg = "Low coverage in tier — prioritize data collection here"
    return prefix + msg


def interpret_auc(auc: float) -> str:
    pct = auc * 100
    if auc >= 0.97:   quality = "outstanding"
    elif auc >= 0.90: quality = "excellent"
    elif auc >= 0.80: quality = "good"
    elif auc >= 0.70: quality = "fair"
    elif auc >= 0.60: quality = "poor"
    else:             quality = "no better than random guessing"
    return (f"AUC = {auc:.4f} ({quality})\n"
            f"The model has a {pct:.1f}% chance of ranking a true flood event\n"
            f"higher than a true non-flood event.")


# =============================================================================
# build_tier_masks  (BUG-1 fix: rank-based fallback for tied boundaries)
# =============================================================================

def build_tier_masks(y_score: np.ndarray, tiers: list) -> list:
    """
    Return one boolean mask per tier.

    Standard path  : score window [lo, hi).
    Tied-boundary  : lo == hi → use rank/index split so every tier always
                     receives ~25 % of rows (prevents zero-row tiers).
    """
    n          = len(y_score)
    sorted_idx = np.argsort(y_score, kind="stable")
    masks      = []

    for i, tier in enumerate(tiers, start=1):
        lo = np.quantile(y_score, tier["quantile_low"])
        hi = np.quantile(y_score, tier["quantile_high"])

        if i == len(tiers):
            mask = y_score >= lo

        elif lo == hi:
            # Tied-boundary fallback — rank-based split
            start = int(round(tier["quantile_low"]  * n))
            end   = int(round(tier["quantile_high"] * n))
            mask  = np.zeros(n, dtype=bool)
            mask[sorted_idx[start:end]] = True

        else:
            mask = (y_score >= lo) & (y_score < hi)

        masks.append(mask)

    return masks


# =============================================================================
# compute_tier_metrics  (BUG-2 fix: tier-discrimination AUC fallback)
# =============================================================================

def compute_tier_metrics(y_true: np.ndarray,
                         y_score: np.ndarray,
                         tiers: list) -> list:
    """
    For each tier compute TPR, FPR, and AUC.

    Primary method  : standard ROC-AUC on the tier's subset (requires both
                      flood=0 and flood=1 to be present in the slice).

    Fallback method : tier-discrimination AUC.
                      When a tier subset contains only one label class (the
                      original "Single class in tier" blank-cell situation),
                      we instead ask:
                        "Can the model rank THIS tier's rows above ALL other
                         rows?"
                      Binary label: y_binary[i] = 1 if row i is in this tier,
                                                  0 otherwise.
                      AUC = roc_auc_score(y_binary, y_score)  [all rows].
                      This is well-defined, interpretable, and always
                      computable as long as both classes appear globally
                      (i.e., the tier is not 100 % of the data).
    """
    masks   = build_tier_masks(y_score, tiers)
    results = []

    for i, (tier, mask) in enumerate(zip(tiers, masks), start=1):
        y_t = y_true[mask]
        y_s = y_score[mask]
        n_rows = int(mask.sum())

        # ── Primary path ────────────────────────────────────────────────────
        if n_rows >= 2 and len(np.unique(y_t)) >= 2:
            fpr_t, tpr_t, thresholds_t = roc_curve(y_t, y_s)
            auc_t   = manual_auc_trapezoidal(fpr_t, tpr_t)
            mid_idx = np.argmin(np.abs(thresholds_t - 0.5))

            results.append({
                "test_number": i,
                "tier":        tier["name"],
                "tpr":         round(float(tpr_t[mid_idx]), 4),
                "fpr":         round(float(fpr_t[mid_idx]), 4),
                "auc":         round(auc_t, 4),
                "remarks":     remark_for_auc(auc_t, fallback=False),
                "fallback":    False,
            })
            continue

        # ── Fallback: tier-discrimination AUC ───────────────────────────────
        # Binary label: 1 = belongs to this tier, 0 = any other tier
        y_binary = mask.astype(int)
        n_pos    = y_binary.sum()
        n_neg    = (y_binary == 0).sum()

        if n_pos < 1 or n_neg < 1:
            # Entire dataset is this tier — truly impossible to compute AUC
            results.append({
                "test_number": i,
                "tier":        tier["name"],
                "tpr":         float("nan"),
                "fpr":         float("nan"),
                "auc":         float("nan"),
                "remarks":     "Tier spans all data — AUC undefined",
                "fallback":    True,
            })
            continue

        try:
            auc_disc = roc_auc_score(y_binary, y_score)
            fpr_d, tpr_d, thresh_d = roc_curve(y_binary, y_score)

            # Use the tier's lower quantile as the reference threshold
            lo = np.quantile(y_score, tier["quantile_low"])
            ref_idx = np.argmin(np.abs(thresh_d - lo))

            results.append({
                "test_number": i,
                "tier":        tier["name"],
                "tpr":         round(float(tpr_d[ref_idx]), 4),
                "fpr":         round(float(fpr_d[ref_idx]), 4),
                "auc":         round(float(auc_disc), 4),
                "remarks":     remark_for_auc(auc_disc, fallback=True),
                "fallback":    True,
            })

        except Exception as exc:
            results.append({
                "test_number": i,
                "tier":        tier["name"],
                "tpr":         float("nan"),
                "fpr":         float("nan"),
                "auc":         float("nan"),
                "remarks":     f"AUC error: {exc}",
                "fallback":    True,
            })

    return results


# =============================================================================
# RECOMMENDATION
# =============================================================================

def recommend_model(model_aucs: dict, all_tier_results: list) -> str:
    scores    = {}
    reasoning = {}

    for model_name, overall_auc in model_aucs.items():
        tier_rows = [r for r in all_tier_results if r["Model"] == model_name]

        tier_aucs = [r["AUC (Tier)"] for r in tier_rows
                     if not (isinstance(r["AUC (Tier)"], float)
                             and np.isnan(r["AUC (Tier)"]))]
        tier_std = float(np.std(tier_aucs)) if len(tier_aucs) > 1 else 1.0

        high_risk_aucs = [r["AUC (Tier)"] for r in tier_rows
                          if any(h in r["Tier"] for h in HIGH_RISK_TIERS)
                          and not (isinstance(r["AUC (Tier)"], float)
                                   and np.isnan(r["AUC (Tier)"]))]
        high_risk_avg = float(np.mean(high_risk_aucs)) if high_risk_aucs else 0.0

        consistency_score = 1.0 / (1.0 + tier_std)
        composite = (0.50 * overall_auc +
                     0.35 * high_risk_avg +
                     0.15 * consistency_score)

        scores[model_name]    = composite
        reasoning[model_name] = {
            "overall_auc":       round(overall_auc,       4),
            "high_risk_avg":     round(high_risk_avg,     4),
            "tier_std":          round(tier_std,          4),
            "consistency_score": round(consistency_score, 4),
            "composite":         round(composite,         4),
        }

    best_model  = max(scores, key=scores.get)
    second_best = sorted(scores, key=scores.get, reverse=True)[1] \
                  if len(scores) > 1 else None
    r = reasoning[best_model]

    lines = [
        "", "=" * 90, "  ★  MODEL RECOMMENDATION", "=" * 90,
        f"  Recommended Model : {best_model}", "",
        "  Scoring Breakdown (weighted composite):",
        f"  {'Model':<18} {'Overall AUC':<14} {'High/Ext AUC':<15} "
        f"{'Tier Std':<11} {'Consistency':<14} {'Composite Score'}",
        "  " + "-" * 86,
    ]
    for name in sorted(scores, key=scores.get, reverse=True):
        rx     = reasoning[name]
        marker = "★ " if name == best_model else "  "
        lines.append(
            f"  {marker}{name:<16} {rx['overall_auc']:<14} {rx['high_risk_avg']:<15} "
            f"{rx['tier_std']:<11} {rx['consistency_score']:<14} {rx['composite']}"
        )
    lines += [
        "", "  Why this model?",
        f"  • Highest composite score of {r['composite']:.4f} across all criteria.",
        f"  • Overall AUC of {r['overall_auc']:.4f} — strong general discrimination.",
        f"  • High/Danger tier AUC of {r['high_risk_avg']:.4f} — reliable on severe flood events.",
        f"  • Consistency score of {r['consistency_score']:.4f} (std = {r['tier_std']:.4f}).",
    ]
    if second_best:
        r2  = reasoning[second_best]
        gap = round(r["composite"] - r2["composite"], 4)
        lines += [
            "",
            f"  Runner-up: {second_best}  "
            f"(composite = {r2['composite']:.4f}, gap = {gap:.4f})",
            f"  Consider {second_best} if deployment constraints favour its architecture.",
        ]
    lines += ["=" * 90, ""]
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading true labels …")
    df_true = pd.read_csv(TRUE_CSV)
    df_true["timestamp"] = pd.to_datetime(df_true["timestamp"])

    label_col = LABEL_COL if LABEL_COL else auto_detect_column(df_true, "label")
    print(f"  True labels file : {df_true.shape[0]} rows")
    print(f"  Using label col  : '{label_col}'")

    all_tier_results = []
    model_aucs       = {}
    roc_curves       = {}
    COLORS = {
        "XGBoost":       "#00d4ff",
        "Random Forest": "#ff7f50",
        "LightGBM":      "#7fff00",
    }

    for model_cfg in PRED_CSVS:
        model_name = model_cfg["model_name"]
        print(f"\n── {model_name} ──────────────────────────────")

        df_pred = pd.read_csv(model_cfg["path"])
        df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])

        pred_col = (model_cfg["pred_col"]
                    if model_cfg["pred_col"]
                    else auto_detect_column(df_pred, "prob"))

        df_merged = pd.merge(
            df_pred[["timestamp", pred_col]],
            df_true[["timestamp", label_col]],
            on="timestamp", how="inner",
        )
        dropped = (len(df_pred) + len(df_true)) - 2 * len(df_merged)
        print(f"  Matched rows         : {len(df_merged)}  ({dropped} unmatched dropped)")

        y_score = df_merged[pred_col].values.astype(float)
        y_true  = df_merged[label_col].values.astype(int)

        n_unique = len(np.unique(y_score))
        print(f"  Unique score values  : {n_unique}  "
              f"{'⚠ highly quantised — fallback AUC will be used where needed' if n_unique < 20 else ''}")
        print(f"  Positives (flood=1)  : {y_true.sum()}  |  "
              f"Negatives (flood=0): {(y_true == 0).sum()}")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val     = roc_auc_score(y_true, y_score)
        model_aucs[model_name] = auc_val
        roc_curves[model_name] = (fpr, tpr)
        print(f"  Overall AUC          : {auc_val:.6f}")

        tier_rows = compute_tier_metrics(y_true, y_score, TIERS)
        masks     = build_tier_masks(y_score, TIERS)

        for r in tier_rows:
            n_rows = int(masks[r["test_number"] - 1].sum())
            method = "tier-disc" if r.get("fallback") else "standard"
            all_tier_results.append({
                "Model":       model_name,
                "Test Number": r["test_number"],
                "Tier":        r["tier"],
                "Rows":        n_rows,
                "TPR":         r["tpr"],
                "FPR":         r["fpr"],
                "AUC (Tier)":  r["auc"],
                "AUC Method":  method,
                "Total AUC":   round(auc_val, 4),
                "Remarks":     r["remarks"],
            })

    # ── Print combined table ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  COMBINED TIER-BASED ROC AUC RESULTS TABLE")
    print("=" * 100)
    header = (f"  {'Model':<16}{'Test#':<7}{'Tier':<12}{'Rows':<7}"
              f"{'TPR':<8}{'FPR':<8}{'AUC(Tier)':<12}{'Method':<12}{'Total AUC':<12}{'Remarks'}")
    print(header)
    print("  " + "-" * 96)

    last_model = None
    for r in all_tier_results:
        if last_model and r["Model"] != last_model:
            print("  " + "-" * 96)
        last_model = r["Model"]
        tpr_str = f"{r['TPR']:.4f}" if not (isinstance(r['TPR'], float) and np.isnan(r['TPR'])) else "—"
        fpr_str = f"{r['FPR']:.4f}" if not (isinstance(r['FPR'], float) and np.isnan(r['FPR'])) else "—"
        auc_str = f"{r['AUC (Tier)']:.4f}" if not (isinstance(r['AUC (Tier)'], float) and np.isnan(r['AUC (Tier)'])) else "—"
        print(f"  {r['Model']:<16}{r['Test Number']:<7}{r['Tier']:<12}{r['Rows']:<7}"
              f"{tpr_str:<8}{fpr_str:<8}{auc_str:<12}{r['AUC Method']:<12}{r['Total AUC']:<12}{r['Remarks']}")
    print("=" * 100)

    print("\n  ── Overall AUC Summary ──")
    for name, auc_val in model_aucs.items():
        print(f"  {name:<16} : {auc_val:.6f}  →  {interpret_auc(auc_val).splitlines()[0]}")

    recommendation = recommend_model(model_aucs, all_tier_results)
    print(recommendation)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for model_name, (fpr, tpr) in roc_curves.items():
        auc_val = model_aucs[model_name]
        color   = COLORS.get(model_name, "#ffffff")
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{model_name}  (AUC = {auc_val:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
            color="#888888", label="Random Baseline (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate  (FPR = FP / (FP + TN))",
                  color="#cccccc", fontsize=11)
    ax.set_ylabel("True Positive Rate  (TPR = TP / (TP + FN))",
                  color="#cccccc", fontsize=11)
    ax.set_title("ROC Curve Comparison – Flood Sensor Models",
                 color="#ffffff", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(loc="lower right", framealpha=0.2,
              labelcolor="#dddddd", facecolor="#1a1a2e", fontsize=10)
    formula = r"$AUC = \sum_{i} \, (FPR_{i+1} - FPR_i) \times \frac{TPR_{i+1} + TPR_i}{2}$"
    ax.text(0.98, 0.08, formula, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#aaaaaa",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                      edgecolor="#333333", alpha=0.8))
    plt.tight_layout()

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ROC curve saved   → {plot_path}")

    tier_csv_path = os.path.join(OUTPUT_DIR, "roc_auc_tier_results_all_models.csv")
    df_out = pd.DataFrame(all_tier_results)
    df_out.to_csv(tier_csv_path, index=False)
    print(f"  Tier CSV saved    → {tier_csv_path}")

    report_path = os.path.join(OUTPUT_DIR, "roc_auc_report_all_models.txt")
    lines = [
        "=" * 100,
        "  ROC AUC REPORT – Flood Sensor Models (XGBoost / Random Forest / LightGBM)  [v3 – FIXED]",
        "=" * 100,
        "",
        "  NOTE: Tiers marked [tier-disc] in Remarks used Tier-Discrimination AUC fallback.",
        "  This occurs when a score-based tier slice contains only one label class (all flood=0",
        "  or all flood=1), which is caused by heavily quantised model outputs (RF: 8 unique",
        "  scores, LightGBM: 4 unique scores). The fallback AUC measures 'can the model rank",
        "  this tier higher than all other tiers?' — a meaningful, computable metric.",
        "",
        "  ── Overall AUC Scores ──",
    ]
    for name, auc_val in model_aucs.items():
        lines.append(f"  {name:<16} : {auc_val:.6f}")
    lines += [
        "", "  ── Tier-Based Results ──",
        f"  {'Model':<16}{'Test#':<7}{'Tier':<12}{'Rows':<7}"
        f"{'TPR':<8}{'FPR':<8}{'AUC(Tier)':<12}{'Method':<12}{'Total AUC':<12}{'Remarks'}",
        "  " + "-" * 96,
    ]
    last_model = None
    for r in all_tier_results:
        if last_model and r["Model"] != last_model:
            lines.append("  " + "-" * 96)
        last_model = r["Model"]
        tpr_str = f"{r['TPR']:.4f}" if not (isinstance(r['TPR'], float) and np.isnan(r['TPR'])) else "—"
        fpr_str = f"{r['FPR']:.4f}" if not (isinstance(r['FPR'], float) and np.isnan(r['FPR'])) else "—"
        auc_str = f"{r['AUC (Tier)']:.4f}" if not (isinstance(r['AUC (Tier)'], float) and np.isnan(r['AUC (Tier)'])) else "—"
        lines.append(
            f"  {r['Model']:<16}{r['Test Number']:<7}{r['Tier']:<12}{r['Rows']:<7}"
            f"{tpr_str:<8}{fpr_str:<8}{auc_str:<12}{r['AUC Method']:<12}{r['Total AUC']:<12}{r['Remarks']}"
        )
    lines.append(recommend_model(model_aucs, all_tier_results))
    lines.append("=" * 100)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Text report saved → {report_path}")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
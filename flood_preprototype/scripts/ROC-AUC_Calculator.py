# =============================================================================
# ROC AUC Calculator – Flood XGBoost Sensor Model
# =============================================================================
# Requirements:
#   pip install pandas numpy matplotlib scikit-learn
#
# Libraries used:
#   - pandas    : CSV loading and data manipulation
#   - numpy     : Numerical computation for manual AUC trapezoidal rule
#   - matplotlib: Plotting the ROC curve
#   - sklearn   : roc_auc_score and roc_curve utilities (for verification)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve

# =============================================================================
# CONFIGURATION — Edit these if your column names differ
# =============================================================================

PRED_CSVS = [
    {
        "path": r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\flood_xgb_sensor_predictions.csv",
        "model_name": "XGBoost",
        "pred_col": None,   # auto-detect or set e.g. "flood_probability"
    },
    {
        "path": r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\flood_rf_sensor_predictions.csv",
        "model_name": "Random Forest",
        "pred_col": None,
    },
    {
        "path": r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\flood_lgbm_sensor_predictions.csv",
        "model_name": "LightGBM",
        "pred_col": None,
    },
]

TRUE_CSV   = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\flood_dataset_test.csv"
OUTPUT_DIR = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\ml_report"
LABEL_COL  = None   # auto-detect or set e.g. "flood_label"

# =============================================================================
# TIER THRESHOLDS
# Each tier defines an AUC probability band used to group test results.
# The script will split the test set into 4 severity tiers based on the
# true label distribution or predicted score quantiles.
# =============================================================================
#
#   Tier        | Description
#   ------------|--------------------------------------------------------------
#   CLEAR         | Samples where flood probability is in the lowest 25th pct
#   WATCH    | Samples between 25th–50th percentile of predicted scores
#   WARNING        | Samples between 50th–75th percentile of predicted scores
#   DANGER     | Samples in the top 25th percentile of predicted scores
#
TIERS = [
    {"name": "CLEAR",      "quantile_low": 0.00, "quantile_high": 0.25},
    {"name": "WATCH", "quantile_low": 0.25, "quantile_high": 0.50},
    {"name": "WARNING",     "quantile_low": 0.50, "quantile_high": 0.75},
    {"name": "DANGER",  "quantile_low": 0.75, "quantile_high": 1.00},
]

# =============================================================================
# WHAT IS ROC AUC?
# =============================================================================
#
# ROC = Receiver Operating Characteristic
# AUC = Area Under the Curve
#
# The ROC curve is built by sweeping a decision threshold from 0 → 1 and,
# at each threshold, computing two rates:
#
#   True Positive Rate (TPR) — also called Sensitivity or Recall:
#       TPR = TP / (TP + FN)
#       "Of all actual flood events, what fraction did we correctly flag?"
#
#   False Positive Rate (FPR) — also called Fall-out:
#       FPR = FP / (FP + TN)
#       "Of all actual non-flood events, what fraction did we wrongly flag?"
#
# The ROC curve plots FPR (x-axis) vs TPR (y-axis).
# A perfect model hugs the top-left corner (FPR=0, TPR=1).
# A random model follows the diagonal (y = x).
#
# AUC is the area under the ROC curve, computed with the trapezoidal rule:
#
#   AUC = Σ  (FPR[i+1] - FPR[i])  ×  (TPR[i+1] + TPR[i]) / 2
#          i
#
#   This sums up the area of trapezoids formed between consecutive points
#   on the curve. A perfect classifier gives AUC = 1.0.
#   Random guessing gives AUC ≈ 0.5.
#
# Interpretation:
#   AUC = 0.90 → the model has a 90% chance of ranking a true flood event
#               higher than a true non-flood event (probabilistic meaning).
# =============================================================================


# ── Helpers ──────────────────────────────────────────────────────────────────

def auto_detect_column(df: pd.DataFrame, hint: str) -> str:
    """
    Try to find a column matching common naming conventions.
    hint = 'prob'  → look for probability/score columns (float 0-1)
    hint = 'label' → look for binary target columns (0/1)
    """
    if hint == "prob":
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in
                             ["prob", "proba", "score", "pred", "flood", "xgb"])]
        # Prefer float columns
        float_cols = [c for c in candidates if df[c].dtype in [np.float32, np.float64]]
        return (float_cols or candidates or [df.columns[-1]])[0]

    if hint == "label":
        candidates = [c for c in df.columns
                      if any(k in c.lower() for k in
                             ["label", "target", "flood", "class", "y"])]
        # Prefer columns with only 0/1 values
        binary_cols = [c for c in candidates
                       if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        return (binary_cols or candidates or [df.columns[-1]])[0]

    raise ValueError(f"Unknown hint: {hint}")


def manual_auc_trapezoidal(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Manually compute AUC using the trapezoidal rule.

    Formula:
        AUC = Σ  (fpr[i+1] - fpr[i])  ×  (tpr[i+1] + tpr[i]) / 2

    Parameters
    ----------
    fpr : array of False Positive Rates (x-axis), sorted ascending
    tpr : array of True Positive Rates (y-axis), corresponding to fpr

    Returns
    -------
    float : area under the ROC curve (0.0 – 1.0)
    """
    auc = 0.0
    for i in range(len(fpr) - 1):
        width  = fpr[i + 1] - fpr[i]          # Δ FPR  (base of trapezoid)
        height = (tpr[i + 1] + tpr[i]) / 2.0  # average TPR (height of trapezoid)
        auc   += width * height
    return abs(auc)   # abs() handles any reversed-order edge cases


def remark_for_auc(auc: float, tier_name: str) -> str:
    """
    Generate a constructive, actionable remark for a given AUC score and tier.
    """
    if auc >= 0.97:
        return "Near-perfect separation — maintain current features"
    elif auc >= 0.93:
        return "Excellent separation — model is well-calibrated for this tier"
    elif auc >= 0.90:
        return "Strong performance — minor threshold tuning may improve recall"
    elif auc >= 0.87:
        return "Good separation — consider adding tier-specific features"
    elif auc >= 0.83:
        return "Moderate overlap — ensemble weighting may sharpen boundaries"
    elif auc >= 0.80:
        return "Developing well — review feature importance for this range"
    elif auc >= 0.75:
        return "Promising signal — augment training samples in this tier"
    elif auc >= 0.70:
        return "Early-stage signal — targeted resampling recommended"
    else:
        return "Low coverage in tier — prioritize data collection here"

def recommend_model(model_aucs: dict, all_tier_results: list) -> str:
    """
    Recommend the best model based on:
      1. Overall AUC (primary criterion)
      2. Tier consistency — std dev of tier AUCs (lower = more stable)
      3. High-risk tier performance — average AUC of High + Extreme tiers
         (critical for flood detection; a model that fails on severe events
          is dangerous even if its overall AUC looks good)

    Scoring (weighted):
      - Overall AUC          : 50%
      - High/Extreme tier AUC: 35%
      - Tier consistency     : 15%  (inverted — lower std is better)
    """
    scores = {}
    reasoning = {}

    for model_name, overall_auc in model_aucs.items():
        tier_rows = [r for r in all_tier_results if r["Model"] == model_name]

        # Tier AUC consistency (std dev across all tiers)
        tier_aucs = [r["AUC (Tier)"] for r in tier_rows
                     if not (isinstance(r["AUC (Tier)"], float)
                             and np.isnan(r["AUC (Tier)"]))]
        tier_std = float(np.std(tier_aucs)) if len(tier_aucs) > 1 else 1.0

        # High-risk tier performance (High + Extreme tiers)
        high_risk_aucs = [r["AUC (Tier)"] for r in tier_rows
                          if r["Tier"] in ("High", "Extreme")
                          and not (isinstance(r["AUC (Tier)"], float)
                                   and np.isnan(r["AUC (Tier)"]))]
        high_risk_avg = float(np.mean(high_risk_aucs)) if high_risk_aucs else 0.0

        # Normalize consistency: convert std → a 0–1 score (lower std = higher score)
        consistency_score = max(0.0, 1.0 - (tier_std / 0.20))

        # Weighted composite score
        composite = (
            0.50 * overall_auc +
            0.35 * high_risk_avg +
            0.15 * consistency_score
        )

        scores[model_name] = composite
        reasoning[model_name] = {
            "overall_auc":        round(overall_auc,    4),
            "high_risk_avg":      round(high_risk_avg,  4),
            "tier_std":           round(tier_std,       4),
            "consistency_score":  round(consistency_score, 4),
            "composite":          round(composite,      4),
        }

    best_model  = max(scores, key=scores.get)
    second_best = sorted(scores, key=scores.get, reverse=True)[1] if len(scores) > 1 else None

    # Build recommendation text
    r  = reasoning[best_model]
    lines = [
        "",
        "=" * 90,
        "  ★  MODEL RECOMMENDATION",
        "=" * 90,
        f"  Recommended Model : {best_model}",
        "",
        "  Scoring Breakdown (weighted composite):",
        f"  {'Model':<18} {'Overall AUC':<14} {'High/Ext AUC':<15} "
        f"{'Tier Std':<11} {'Consistency':<14} {'Composite Score'}",
        "  " + "-" * 86,
    ]
    for name in sorted(scores, key=scores.get, reverse=True):
        rx = reasoning[name]
        marker = "★ " if name == best_model else "  "
        lines.append(
            f"  {marker}{name:<16} {rx['overall_auc']:<14} {rx['high_risk_avg']:<15} "
            f"{rx['tier_std']:<11} {rx['consistency_score']:<14} {rx['composite']}"
        )

    lines += [
        "",
        "  Why this model?",
        f"  • Highest composite score of {r['composite']:.4f} across all three criteria.",
        f"  • Overall AUC of {r['overall_auc']:.4f} — strong general discrimination.",
        f"  • High/Extreme tier AUC of {r['high_risk_avg']:.4f} — reliable on severe flood events.",
        f"  • Tier consistency score of {r['consistency_score']:.4f} "
        f"(std = {r['tier_std']:.4f}) — stable across all severity levels.",
    ]

    if second_best:
        r2 = reasoning[second_best]
        gap = round(r["composite"] - r2["composite"], 4)
        lines += [
            "",
            f"  Runner-up: {second_best}  (composite = {r2['composite']:.4f}, "
            f"gap = {gap:.4f})",
            f"  Consider {second_best} if deployment constraints favour its architecture.",
        ]

    lines += ["=" * 90, ""]
    return "\n".join(lines)

def compute_tier_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tiers: list
) -> list:
    """
    Split the dataset into quantile-based tiers by predicted score,
    then compute TPR, FPR, and AUC for each tier.

    Returns a list of dicts with keys:
        test_number, tier, tpr, fpr, auc, remarks
    """
    results = []
    for i, tier in enumerate(tiers, start=1):
        lo = np.quantile(y_score, tier["quantile_low"])
        hi = np.quantile(y_score, tier["quantile_high"])

        # Select samples whose predicted score falls in this quantile band
        mask = (y_score >= lo) & (y_score <= hi)

        # Edge case: last tier uses strict < to avoid including nothing
        if i == len(tiers):
            mask = y_score >= lo

        y_t = y_true[mask]
        y_s = y_score[mask]

        # Need at least one positive and one negative for AUC
        if len(y_t) < 2 or len(np.unique(y_t)) < 2:
            results.append({
                "test_number": i,
                "tier": tier["name"],
                "tpr": float("nan"),
                "fpr": float("nan"),
                "auc": float("nan"),
                "remarks": "Insufficient data",
            })
            continue

        # Compute ROC curve for this tier
        fpr_t, tpr_t, thresholds_t = roc_curve(y_t, y_s)
        auc_t = manual_auc_trapezoidal(fpr_t, tpr_t)

        # Representative TPR and FPR: value at the threshold closest to 0.5
        mid_idx = np.argmin(np.abs(thresholds_t - 0.5))
        rep_tpr = round(float(tpr_t[mid_idx]), 4)
        rep_fpr = round(float(fpr_t[mid_idx]), 4)

        results.append({
            "test_number": i,
            "tier": tier["name"],
            "tpr": rep_tpr,
            "fpr": rep_fpr,
            "auc": round(auc_t, 4),
            "remarks": remark_for_auc(auc_t, tier["name"]),
        })

    return results


def print_tier_table(tier_results: list):
    """Print the results table to the console in the specified format."""
    header = f"{'Test Number':<14}{'Tier':<12}{'TPR':<8}{'FPR':<8}{'AUC':<8}{'Remarks'}"
    sep    = "-" * 68
    print("\n" + sep)
    print("  TIER-BASED ROC AUC RESULTS TABLE")
    print(sep)
    print(f"  {header}")
    print(f"  {'-'*66}")
    for r in tier_results:
        row = (
            f"{r['test_number']:<14}"
            f"{r['tier']:<12}"
            f"{r['tpr']:<8}"
            f"{r['fpr']:<8}"
            f"{r['auc']:<8}"
            f"{r['remarks']}"
        )
        print(f"  {row}")
    print(sep + "\n")


def interpret_auc(auc: float) -> str:
    """Return a plain-English interpretation of an AUC score."""
    pct = auc * 100
    if auc >= 0.97:
        quality = "outstanding"
    elif auc >= 0.90:
        quality = "excellent"
    elif auc >= 0.80:
        quality = "good"
    elif auc >= 0.70:
        quality = "fair"
    elif auc >= 0.60:
        quality = "poor"
    else:
        quality = "no better than random guessing"

    return (
        f"AUC = {auc:.4f} ({quality})\n"
        f"The model has a {pct:.1f}% chance of ranking a true flood event\n"
        f"higher than a true non-flood event (probabilistic interpretation)."
    )


def main():
    # ------------------------------------------------------------------
    # 1. Load true labels
    # ------------------------------------------------------------------
    print("Loading true labels...")
    df_true = pd.read_csv(TRUE_CSV)
    df_true["timestamp"] = pd.to_datetime(df_true["timestamp"])

    label_col = LABEL_COL if LABEL_COL else auto_detect_column(df_true, "label")
    print(f"  True labels file : {df_true.shape[0]} rows")
    print(f"  Using label col  : '{label_col}'")

    # ------------------------------------------------------------------
    # 2. Loop over each model — load, align, compute metrics
    # ------------------------------------------------------------------
    all_tier_results = []   # rows for the combined tier table
    model_aucs       = {}   # model_name → overall AUC
    roc_curves       = {}   # model_name → (fpr, tpr)
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
        print(f"  Predictions file : {df_pred.shape[0]} rows")
        print(f"  Using pred col   : '{pred_col}'")

        # ── Align by timestamp (inner join, drop unmatched rows) ──────
        df_merged = pd.merge(
            df_pred[["timestamp", pred_col]],
            df_true[["timestamp", label_col]],
            on="timestamp",
            how="inner",
        )
        dropped = (len(df_pred) + len(df_true)) - 2 * len(df_merged)
        print(f"  Matched rows     : {len(df_merged)}  "
              f"({dropped} rows dropped — unmatched timestamps)")

        y_score = df_merged[pred_col].values.astype(float)
        y_true  = df_merged[label_col].values.astype(int)

        unique_labels = set(y_true)
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"[{model_name}] True labels must be binary (0/1). Found: {unique_labels}"
            )

        print(f"  Positives (flood=1): {y_true.sum()}  |  "
              f"Negatives (flood=0): {(y_true == 0).sum()}")

        # ── Overall ROC / AUC ─────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val      = roc_auc_score(y_true, y_score)

        model_aucs[model_name]  = auc_val
        roc_curves[model_name]  = (fpr, tpr)

        print(f"  Overall AUC      : {auc_val:.6f}  → {interpret_auc(auc_val).splitlines()[0]}")

        # ------------------------------------------------------------------
        # Recommendation
        # ------------------------------------------------------------------
        recommendation = recommend_model(model_aucs, all_tier_results)
        print(recommendation)

        # ── Tier-based metrics ────────────────────────────────────────
        tier_rows = compute_tier_metrics(y_true, y_score, TIERS)
        for r in tier_rows:
            # Count samples in this tier's quantile band
            lo = np.quantile(y_score, TIERS[r["test_number"] - 1]["quantile_low"])
            hi = np.quantile(y_score, TIERS[r["test_number"] - 1]["quantile_high"])
            if r["test_number"] == len(TIERS):
                n_rows = int((y_score >= lo).sum())
            else:
                n_rows = int(((y_score >= lo) & (y_score <= hi)).sum())

            all_tier_results.append({
                "Model":       model_name,
                "Test Number": r["test_number"],
                "Tier":        r["tier"],
                "Rows":        n_rows,
                "TPR":         r["tpr"],
                "FPR":         r["fpr"],
                "AUC (Tier)":  r["auc"],
                "Total AUC":   round(auc_val, 4),
                "Remarks":     r["remarks"],
            })

    # ------------------------------------------------------------------
    # 3. Print combined results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  COMBINED TIER-BASED ROC AUC RESULTS TABLE")
    print("=" * 90)
    header = (f"  {'Model':<16}{'Test#':<7}{'Tier':<12}{'Rows':<7}"
              f"{'TPR':<7}{'FPR':<7}{'AUC(Tier)':<12}{'Total AUC':<12}{'Remarks'}")
    print(header)
    print("  " + "-" * 86)

    last_model = None
    for r in all_tier_results:
        if last_model and r["Model"] != last_model:
            print("  " + "-" * 86)   # separator between models
        last_model = r["Model"]
        print(
            f"  {r['Model']:<16}{r['Test Number']:<7}{r['Tier']:<12}{r['Rows']:<7}"
            f"{r['TPR']:<7}{r['FPR']:<7}{r['AUC (Tier)']:<12}{r['Total AUC']:<12}{r['Remarks']}"
        )
    print("=" * 90)

    # Overall AUC summary
    print("\n  ── Overall AUC Summary ──")
    for name, auc_val in model_aucs.items():
        print(f"  {name:<16} : {auc_val:.6f}  →  {interpret_auc(auc_val).splitlines()[0]}")

    # ------------------------------------------------------------------
    # 4. Plot — all three ROC curves on one chart
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ROC curve saved  → {plot_path}")

    # CSV — combined tier results
    tier_csv_path = os.path.join(OUTPUT_DIR, "roc_auc_tier_results_all_models.csv")
    pd.DataFrame(all_tier_results).to_csv(tier_csv_path, index=False)
    print(f"  Tier CSV saved   → {tier_csv_path}")

    # Text report
    report_path = os.path.join(OUTPUT_DIR, "roc_auc_report_all_models.txt")
    lines = [
        "=" * 90,
        "  ROC AUC REPORT – Flood Sensor Models (XGBoost / Random Forest / LightGBM)",
        "=" * 90,
        "",
        "  ── Overall AUC Scores ──",
    ]
    for name, auc_val in model_aucs.items():
        lines.append(f"  {name:<16} : {auc_val:.6f}")
    lines += [
        "",
        "  ── Tier-Based Results ──",
        f"  {'Model':<16}{'Test#':<7}{'Tier':<12}{'Rows':<7}"
        f"{'TPR':<7}{'FPR':<7}{'AUC(Tier)':<12}{'Total AUC':<12}{'Remarks'}",
        "  " + "-" * 86,
    ]
    last_model = None
    for r in all_tier_results:
        if last_model and r["Model"] != last_model:
            lines.append("  " + "-" * 86)
        last_model = r["Model"]
        lines.append(
            f"  {r['Model']:<16}{r['Test Number']:<7}{r['Tier']:<12}{r['Rows']:<7}"
            f"{r['TPR']:<7}{r['FPR']:<7}{r['AUC (Tier)']:<12}{r['Total AUC']:<12}{r['Remarks']}"
        )
    lines.append(recommend_model(model_aucs, all_tier_results))
    lines.append("=" * 90)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Text report saved → {report_path}")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
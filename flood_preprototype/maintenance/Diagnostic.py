"""
diagnose_flood_model.py
=======================
Diagnostic script for the Rapid Relay flood prediction pipeline.

Runs four diagnostic panels:

    1. Feature Distribution Shift
       KDE + boxplot of each feature across Train / Val / Test splits.
       Flags features whose test mean deviates > 1 std from train mean.

    2. Prediction vs Label Inspection
       Plots flood_probability over time for Val and Test, overlaid
       with true flood labels. Helps identify if the model is inverting,
       collapsing to 0.5, or just uncertain.

    3. Label Distribution Over Time
       Rolling flood rate across the full dataset. Reveals if flood
       frequency changed significantly between splits — which would
       explain a label drift problem.

    4. Correlation Heatmap per Split
       Spearman correlation between each feature and the flood label,
       computed separately for Train / Val / Test.
       A feature with high train correlation but near-zero test
       correlation is a drift indicator.

All plots saved to ../diagnostics/
"""

import os
import sys
import warnings
import joblib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
_ML_PIPELINE  = os.path.join(_PROJECT_ROOT, "ml_pipeline")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, _ML_PIPELINE)

from feature_engineering import SENSOR_FEATURE_COLUMNS, FULL_FEATURE_COLUMNS

# ===========================================================================
# CONFIG — must match training scripts
# ===========================================================================

DATA_FILE    = r"..\data\flood_dataset.csv"
MODEL_FILE   = r"..\model\flood_xgb_sensor.pkl"   # sensor model for predictions
OUTPUT_DIR   = r"..\diagnostics"

LABEL_COLUMN = "flood_label"
TRAIN_END    = "2022-12-31"
VAL_END      = "2024-12-31"

# ===========================================================================


def sep(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# Load dataset and split
# ---------------------------------------------------------------------------

def load_and_split(path, feat_cols):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index().dropna(how="all").drop_duplicates()

    tz = df.index.tz
    train_end = pd.Timestamp(TRAIN_END, tz=tz)
    val_end   = pd.Timestamp(VAL_END,   tz=tz)

    present   = [c for c in feat_cols if c in df.columns]
    missing   = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  Skipping missing columns: {missing}")

    train = df[df.index <= train_end]
    val   = df[(df.index > train_end) & (df.index <= val_end)]
    test  = df[df.index > val_end]

    print(f"  Train : {len(train):>4} rows  "
          f"({train.index.min().date()} -> {train.index.max().date()})  "
          f"flood={int(train[LABEL_COLUMN].sum())}")
    print(f"  Val   : {len(val):>4} rows  "
          f"({val.index.min().date()} -> {val.index.max().date()})  "
          f"flood={int(val[LABEL_COLUMN].sum())}")
    print(f"  Test  : {len(test):>4} rows  "
          f"({test.index.min().date()} -> {test.index.max().date()})  "
          f"flood={int(test[LABEL_COLUMN].sum())}")

    return df, train, val, test, present


# ---------------------------------------------------------------------------
# DIAGNOSTIC 1 — Feature Distribution Shift
# ---------------------------------------------------------------------------

def diag_feature_distributions(train, val, test, feat_cols, output_dir):
    sep("Diagnostic 1 — Feature Distribution Shift")

    splits = {
        "Train\n(2017–2022)": train,
        "Val\n(2023–2024)":   val,
        "Test\n(2025–2026)":  test,
    }
    colors = {"Train\n(2017–2022)": "#4C72B0",
               "Val\n(2023–2024)":   "#DD8452",
               "Test\n(2025–2026)":  "#55A868"}

    n = len(feat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes = axes.flatten()

    flagged = []

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        train_vals = train[col].dropna()
        train_mean = train_vals.mean()
        train_std  = train_vals.std()

        for label, df_split in splits.items():
            vals = df_split[col].dropna()
            if len(vals) < 2:
                continue
            vals.plot.kde(ax=ax, label=label, color=colors[label], linewidth=1.8)
            ax.axvline(vals.mean(), color=colors[label], linestyle="--",
                       linewidth=1, alpha=0.7)

        # Flag drift: test mean > 1 std from train mean
        test_vals = test[col].dropna()
        if len(test_vals) > 0 and train_std > 0:
            shift = abs(test_vals.mean() - train_mean) / train_std
            drift_flag = "  ⚠️ DRIFT" if shift > 1.0 else ""
            if shift > 1.0:
                flagged.append((col, shift))
        else:
            drift_flag = ""
            shift = 0.0

        ax.set_title(f"{col}{drift_flag}", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

        # Print summary stats
        print(f"\n  {col}")
        for label, df_split in splits.items():
            vals = df_split[col].dropna()
            if len(vals) == 0:
                print(f"    {label.replace(chr(10),' '):<22} : NO DATA")
            else:
                print(f"    {label.replace(chr(10),' '):<22} : "
                      f"mean={vals.mean():>8.3f}  std={vals.std():>7.3f}  "
                      f"min={vals.min():>8.3f}  max={vals.max():>8.3f}")
        if shift > 0:
            marker = "⚠️ " if shift > 1.0 else "  "
            print(f"    {marker}Test drift from train mean : {shift:.2f} std")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distribution: Train / Val / Test\n"
                 "(dashed lines = split means; ⚠️ = test mean > 1σ from train)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, "diag1_feature_distributions.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")

    if flagged:
        print(f"\n  ⚠️  DRIFTED FEATURES (test mean > 1σ from train):")
        for col, shift in sorted(flagged, key=lambda x: -x[1]):
            print(f"    {col:<40} shift = {shift:.2f} σ")
    else:
        print("\n  ✅  No severe feature drift detected (all shifts < 1σ).")

    return flagged


# ---------------------------------------------------------------------------
# DIAGNOSTIC 2 — Prediction vs Label Over Time
# ---------------------------------------------------------------------------

def diag_prediction_vs_label(df, train, val, test, feat_cols,
                              model, output_dir):
    sep("Diagnostic 2 — Prediction vs Label Over Time")

    available = [c for c in feat_cols if c in df.columns]

    # Predict on val + test (unseen data)
    for split_name, split_df in [("Val (2023–2024)", val),
                                  ("Test (2025–2026)", test)]:
        if len(split_df) == 0:
            print(f"  {split_name}: no data, skipping.")
            continue

        X     = split_df[available].values
        probs = model.predict_proba(X)[:, 1]
        y     = split_df[LABEL_COLUMN].values
        dates = split_df.index

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        # Top: probability line + flood label shading
        ax = axes[0]
        ax.plot(dates, probs, color="steelblue", linewidth=1.5,
                label="Flood Probability", zorder=3)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="Threshold (0.50)")

        # Shade actual flood periods
        for j, (ts, prob) in enumerate(zip(dates, y)):
            if prob == 1:
                ax.axvspan(dates[j], dates[min(j + 1, len(dates) - 1)],
                           alpha=0.20, color="crimson", zorder=1)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Flood Probability")
        ax.set_title(f"{split_name} — Prediction vs True Label\n"
                     f"(red shading = actual flood periods)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Bottom: scatter — correct vs wrong predictions
        ax2 = axes[1]
        y_pred = (probs >= 0.5).astype(int)
        correct = y_pred == y
        wrong   = ~correct

        ax2.scatter(dates[correct], probs[correct], c="green", s=40,
                    alpha=0.7, label="Correct", zorder=3)
        ax2.scatter(dates[wrong],   probs[wrong],   c="red",   s=60,
                    alpha=0.8, marker="X", label="Wrong", zorder=4)
        ax2.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Flood Probability")
        ax2.set_xlabel("Date")
        ax2.set_title(f"Prediction Errors  (correct={correct.sum()}  "
                      f"wrong={wrong.sum()}  accuracy={correct.mean():.1%})")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        safe = split_name.lower().replace(" ", "_").replace("(","").replace(")","").replace("–","")
        out  = os.path.join(output_dir, f"diag2_pred_vs_label_{safe}.png")
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"  {split_name} saved → {out}")

        # Confusion breakdown
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        tn = ((y_pred == 0) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        print(f"  {split_name}:")
        print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"    Prob when label=1 : mean={probs[y==1].mean():.3f}  "
              f"std={probs[y==1].std():.3f}")
        print(f"    Prob when label=0 : mean={probs[y==0].mean():.3f}  "
              f"std={probs[y==0].std():.3f}")

        separation = probs[y==1].mean() - probs[y==0].mean()
        print(f"    Probability separation (flood - no-flood) : {separation:+.3f}")
        if separation < 0:
            print(f"    ❌ INVERSION: model assigns HIGHER prob to non-flood rows!")
        elif separation < 0.05:
            print(f"    ⚠️  Near-zero separation: model cannot distinguish classes.")
        else:
            print(f"    ✅  Positive separation: model direction is correct.")


# ---------------------------------------------------------------------------
# DIAGNOSTIC 3 — Label (Flood Rate) Over Time
# ---------------------------------------------------------------------------

def diag_label_drift(df, output_dir):
    sep("Diagnostic 3 — Flood Rate Over Time")

    # Annual flood rate
    annual = df[LABEL_COLUMN].resample("YE").agg(["sum", "count"])
    annual["rate"] = annual["sum"] / annual["count"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: annual flood rate bar chart
    ax = axes[0]
    years  = annual.index.year
    rates  = annual["rate"].values
    counts = annual["sum"].values
    bars   = ax.bar(years, rates, color="steelblue", alpha=0.8)
    ax.axhline(rates.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"Overall mean ({rates.mean():.1%})")

    # Colour train/val/test bars differently
    for bar, year in zip(bars, years):
        if year <= 2022:
            bar.set_color("#4C72B0")
        elif year <= 2024:
            bar.set_color("#DD8452")
        else:
            bar.set_color("#55A868")

    for bar, rate, count in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.0%}\n(n={int(count)})", ha="center", fontsize=8)

    ax.set_ylabel("Flood Rate")
    ax.set_title("Annual Flood Rate  "
                 "(blue=train 2017–2022  orange=val 2023–2024  green=test 2025–2026)")
    ax.set_ylim(0, 1)
    ax.legend()

    # Bottom: cumulative flood label over time
    ax2 = axes[1]
    cumulative_rate = df[LABEL_COLUMN].expanding().mean()
    ax2.plot(df.index, cumulative_rate, color="steelblue", linewidth=1.5)
    ax2.axvline(pd.Timestamp(TRAIN_END, tz=df.index.tz),
                color="#DD8452", linestyle="--", linewidth=1.5, label="Train/Val split")
    ax2.axvline(pd.Timestamp(VAL_END, tz=df.index.tz),
                color="#55A868", linestyle="--", linewidth=1.5, label="Val/Test split")
    ax2.set_ylabel("Cumulative Flood Rate")
    ax2.set_xlabel("Date")
    ax2.set_title("Cumulative Flood Rate Over Time  (drift = slope change after split lines)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "diag3_label_drift.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved → {out}")

    # Print annual stats
    print(f"\n  Annual flood rates:")
    for year, row in annual.iterrows():
        marker = ("🔵 Train" if year.year <= 2022
                  else "🟠 Val  " if year.year <= 2024
                  else "🟢 Test ")
        print(f"    {year.year}  {marker}  floods={int(row['sum']):>3}  "
              f"total={int(row['count']):>3}  rate={row['rate']:.1%}")


# ---------------------------------------------------------------------------
# DIAGNOSTIC 4 — Feature-Label Correlation per Split
# ---------------------------------------------------------------------------

def diag_correlation_shift(train, val, test, feat_cols, output_dir):
    sep("Diagnostic 4 — Feature–Label Spearman Correlation per Split")

    splits = {
        "Train\n2017–2022": train,
        "Val\n2023–2024":   val,
        "Test\n2025–2026":  test,
    }

    corr_data = {}
    for split_name, df_split in splits.items():
        corrs = {}
        for col in feat_cols:
            if col not in df_split.columns:
                corrs[col] = 0.0
                continue
            vals  = df_split[[col, LABEL_COLUMN]].dropna()
            if len(vals) < 5 or vals[col].std() == 0:
                corrs[col] = 0.0
            else:
                r, _ = stats.spearmanr(vals[col], vals[LABEL_COLUMN])
                corrs[col] = r
        corr_data[split_name] = corrs

    corr_df = pd.DataFrame(corr_data)

    # Flag features where sign flips between train and test
    train_col = "Train\n2017–2022"
    test_col  = "Test\n2025–2026"
    sign_flips = []
    for feat in feat_cols:
        t = corr_df.loc[feat, train_col]
        s = corr_df.loc[feat, test_col]
        if abs(t) > 0.05 and abs(s) > 0.05 and np.sign(t) != np.sign(s):
            sign_flips.append(feat)

    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_cols) * 0.55)))
    x = np.arange(len(feat_cols))
    width = 0.25
    colors_list = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (split_name, color) in enumerate(zip(splits.keys(), colors_list)):
        vals = [corr_df.loc[f, split_name] for f in feat_cols]
        ax.barh(x + i * width, vals, width, label=split_name.replace("\n", " "),
                color=color, alpha=0.85)

    ax.set_yticks(x + width)
    ax.set_yticklabels(feat_cols, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman Correlation with Flood Label")
    ax.set_title("Feature–Label Correlation by Split\n"
                 "(sign flip between Train and Test = relationship reversed)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "diag4_correlation_shift.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved → {out}")

    # Print table
    print(f"\n  {'Feature':<40} {'Train':>8}  {'Val':>8}  {'Test':>8}  Note")
    print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*8}  ----")
    for feat in feat_cols:
        t = corr_df.loc[feat, train_col]
        v = corr_df.loc[feat, "Val\n2023–2024"]
        s = corr_df.loc[feat, test_col]
        note = "🔄 SIGN FLIP" if feat in sign_flips else ""
        print(f"  {feat:<40} {t:>+8.3f}  {v:>+8.3f}  {s:>+8.3f}  {note}")

    if sign_flips:
        print(f"\n  🔄 Features with SIGN FLIP (train vs test):")
        for f in sign_flips:
            print(f"    {f}")
        print(f"\n  A sign flip means the feature's relationship with flooding")
        print(f"  reversed between training and test periods. This is a strong")
        print(f"  indicator of distribution shift or label drift.")
    else:
        print(f"\n  ✅  No sign flips detected.")

    return corr_df, sign_flips


# ---------------------------------------------------------------------------
# DIAGNOSTIC 5 — Quick Summary / Verdict
# ---------------------------------------------------------------------------

def print_verdict(flagged_drift, sign_flips, output_dir):
    sep("VERDICT — Root Cause Assessment")

    issues = []

    if flagged_drift:
        issues.append(("Feature Distribution Shift",
                        f"{len(flagged_drift)} features drifted > 1σ: "
                        + ", ".join(f[0] for f in flagged_drift[:3])))

    if sign_flips:
        issues.append(("Feature–Label Relationship Reversal",
                        f"Sign flips on: " + ", ".join(sign_flips[:3])))

    if not issues:
        print("\n  No severe structural issues detected.")
        print("  Low test AUC likely due to small test set size or")
        print("  genuine uncertainty in flood prediction at this lead time.")
    else:
        print(f"\n  Found {len(issues)} issue(s):\n")
        for i, (title, detail) in enumerate(issues, 1):
            print(f"  {i}. {title}")
            print(f"     {detail}\n")

    print("  Recommended actions:")
    print("  ─────────────────────────────────────────────────────")
    if flagged_drift:
        print("  → Feature drift detected:")
        print("    - Standardize features using train-set mean/std (StandardScaler)")
        print("    - Add year or day-of-year as a feature to help the model")
        print("      account for temporal trends")
        print("    - Consider dropping drifted features from the deployed model")
    if sign_flips:
        print("  → Sign flips detected:")
        print("    - These features are actively harming test performance")
        print("    - Remove them and retrain; check if AUC improves")
        print("    - Investigate if SAR acquisition geometry changed post-2024")
    print("  → General:")
    print("    - Try training on 2017–2024 (train+val) and test on 2025–2026")
    print("    - Run RF and LGBM scripts — compare test AUC across all 3 models")
    print("    - Check raw SAR backscatter values in sentinel1_timeseries.csv")
    print("      for any step-change discontinuity around 2025")
    print(f"\n  All diagnostic plots saved to: {output_dir}")
    sep()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sep("Flood Model Diagnostics — Rapid Relay")
    print(f"  Data  : {DATA_FILE}")
    print(f"  Model : {MODEL_FILE}")
    print(f"  Out   : {OUTPUT_DIR}")

    # Load model
    sep("Loading Model")
    if not os.path.exists(MODEL_FILE):
        sys.exit(f"\n  ERROR: Model not found at {MODEL_FILE}\n"
                 f"  Run XGB_train_flood_model.py first.")
    artifact     = joblib.load(MODEL_FILE)
    model        = artifact["model"]
    feature_cols = artifact["feature_columns"]
    print(f"  Loaded : {type(model).__name__}  ({len(feature_cols)} features)")

    # Use FULL feature columns for loading, sensor subset for model
    sep("Loading & Splitting Data")
    df, train, val, test, all_feat_cols = load_and_split(
        DATA_FILE, FULL_FEATURE_COLUMNS
    )

    # Diagnostics
    flagged_drift = diag_feature_distributions(
        train, val, test, all_feat_cols, OUTPUT_DIR
    )
    diag_prediction_vs_label(
        df, train, val, test, feature_cols, model, OUTPUT_DIR
    )
    diag_label_drift(df, OUTPUT_DIR)
    corr_df, sign_flips = diag_correlation_shift(
        train, val, test, all_feat_cols, OUTPUT_DIR
    )
    print_verdict(flagged_drift, sign_flips, OUTPUT_DIR)


if __name__ == "__main__":
    main()
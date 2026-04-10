import pandas as pd

# =========================================================
# CONFIG
# =========================================================
input_path = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions\speedtest_predictions.csv"
output_path = input_path.replace(".csv", "_pst_normalized.csv")

ASSUME_NAIVE_AS = "UTC"  # or "Asia/Manila"
TARGET_TZ = "Asia/Manila"

# =========================================================
# LOAD CSV
# =========================================================
df = pd.read_csv(input_path)

# =========================================================
# DETERMINE BASE DATE
# =========================================================
def extract_base_date(df, columns):
    for col in columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.dropna()
            if not valid.empty:
                return valid.iloc[0].date()
    return pd.Timestamp.now().date()

timestamp_cols = ["sensor_timestamp", "prediction_created_at", "row_created_at"]
base_date = extract_base_date(df, timestamp_cols)

# =========================================================
# NORMALIZATION FUNCTION
# =========================================================
def normalize_timestamp(val):
    if pd.isna(val):
        return pd.NaT

    val_str = str(val).strip()

    # Handle time-only values
    if len(val_str) <= 8 and ":" in val_str:
        val_str = f"{base_date} {val_str}"

    ts = pd.to_datetime(val_str, errors="coerce")

    if pd.isna(ts):
        return pd.NaT

    # Handle timezone
    if ts.tzinfo is None:
        if ASSUME_NAIVE_AS == "UTC":
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_localize(TARGET_TZ)

    # Convert to PST
    ts = ts.tz_convert(TARGET_TZ)

    # Round to seconds
    ts = ts.round("s")

    return ts

# =========================================================
# APPLY NORMALIZATION
# =========================================================
for col in timestamp_cols:
    if col in df.columns:
        df[col] = df[col].apply(normalize_timestamp)

# =========================================================
# COMPUTE TIME DIFFERENCE (row_created_at - sensor_timestamp)
# =========================================================
if "sensor_timestamp" in df.columns and "row_created_at" in df.columns:
    df["time_diff"] = df["row_created_at"] - df["sensor_timestamp"]

    # Convert to seconds
    df["time_diff_seconds"] = df["time_diff"].dt.total_seconds()

# =========================================================
# FORMAT TIMESTAMPS TO ISO 8601 (+08:00)
# =========================================================
for col in timestamp_cols:
    if col in df.columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        df[col] = df[col].str.replace(r"(\+|\-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)

# =========================================================
# ADD TEST NUMBER COLUMN (LEFTMOST)
# =========================================================
df.insert(0, "test_no", range(1, len(df) + 1))

# =========================================================
# MOVE time_diff_seconds TO RIGHTMOST
# =========================================================
if "time_diff_seconds" in df.columns:
    col = df.pop("time_diff_seconds")
    df["sensor_minus_row_created_at_sec"] = col

# =========================================================
# OPTIONAL: KEEP ONLY RELEVANT COLUMNS
# =========================================================
final_cols = ["test_no"] + \
             [col for col in timestamp_cols if col in df.columns] + \
             (["sensor_minus_row_created_at_sec"] if "sensor_minus_row_created_at_sec" in df.columns else [])

df = df[final_cols]

# =========================================================
# SAVE OUTPUT
# =========================================================
df.to_csv(output_path, index=False)

print(f"✅ Final dataset saved to: {output_path}")
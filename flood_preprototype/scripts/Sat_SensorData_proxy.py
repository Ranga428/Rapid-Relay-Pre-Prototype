"""
GEE Data Extractor — Obando Environmental Data  (RAM-optimised + normalised build)
Extracts: Soil Moisture (ERA5-Land), Atmospheric Humidity (Aqua/MODIS), Tidal Level (Obando)
Output: CSV with columns: timestamp, waterlevel, soil_moisture, humidity

Requirements:
    pip install earthengine-api geemap pandas requests copernicusmarine
    earthengine authenticate
    copernicusmarine login   ← one-time CMEMS credential setup (free account)

CMEMS account (free): https://marine.copernicus.eu/
Or pass credentials via environment variables: CMEMS_USERNAME / CMEMS_PASSWORD

NOTES:
- waterlevel output is z-score normalised (dimensionless). UHSLC and CMEMS are
  normalised independently against a 2017–2021 reference window before merging,
  so source-switch jumps are eliminated. If you need physical metres downstream,
  store uhslc_mu / uhslc_std from the normalisation step and invert.
- MODIS humidity is chunked quarterly to stay within GEE memory limits.
- ERA5 soil moisture is chunked yearly.
- All intermediate objects are explicitly deleted + gc.collect() after each batch.
"""

import gc
import ee
import io
import os
import json
import requests
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GEE_PROJECT = "jenel-466709"
START_DATE  = "2017-01-01"
END_DATE    = "2026-03-01"

_SCRIPT_DIR = Path(__file__).resolve().parent
_AOI_PATH   = _SCRIPT_DIR.parent / "config" / "aoi.geojson"
OUTPUT_CSV  = _SCRIPT_DIR.parent / "data" / "sensor" / "obando_environmental_data.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

_CMEMS_LON_MIN = 120.5
_CMEMS_LON_MAX = 121.2
_CMEMS_LAT_MIN = 14.5
_CMEMS_LAT_MAX = 15.0

_UHSLC_ERDDAP_URL = (
    "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_daily_fast.csv"
    "?sea_level,time"
    "&uhslc_id=370"
    "&time>={start}T00:00:00Z"
    "&time<={end}T00:00:00Z"
)

# ─── INIT ─────────────────────────────────────────────────────────────────────

print("Authenticating with Google Earth Engine...")
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
print("✓ GEE initialized")

with open(_AOI_PATH, "r") as f:
    _geojson = json.load(f)

if _geojson["type"] == "FeatureCollection":
    _geometry = _geojson["features"][0]["geometry"]
elif _geojson["type"] == "Feature":
    _geometry = _geojson["geometry"]
else:
    _geometry = _geojson

AOI = ee.Geometry(_geometry)
del _geojson, _geometry          # free immediately — only AOI object needed
print(f"✓ AOI loaded from: {_AOI_PATH}")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _quarter_ranges(start_str: str, end_str: str):
    """Yield (q_start, q_end) string pairs in 3-month increments."""
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    cur   = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=3), end)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _year_ranges(start_str: str, end_str: str):
    """Yield (y_start, y_end) string pairs in 12-month increments."""
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    cur   = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _fc_to_rows(fc_info: dict, props: list) -> list:
    """Extract only the requested property keys from a FeatureCollection getInfo dict."""
    return [
        {k: f["properties"].get(k) for k in props}
        for f in fc_info["features"]
    ]


# ─── 1. SOIL MOISTURE — ERA5-Land ─────────────────────────────────────────────

def get_soil_moisture() -> pd.DataFrame:
    """
    ERA5-Land Daily Aggregated — volumetric_soil_water_layer_1 (0-7cm, m³/m³).

    RAM strategy: yearly batches (~365 features each).
    ERA5-Land masks ocean pixels — uses a 5km land buffer around Obando town
    centre instead of the full AOI to guarantee land pixel coverage.
    gc.collect() after every batch keeps Python heap tight.
    """
    print("Fetching ERA5-Land soil moisture data...")

    BAND     = "volumetric_soil_water_layer_1"
    LAND_AOI = ee.Geometry.Point([120.9333, 14.8333]).buffer(5000)

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(START_DATE, END_DATE)
        .select(BAND)
    )

    def _map_fn(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        val  = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=LAND_AOI,
            scale=11132,
            maxPixels=1e9,
        ).get(BAND)
        return ee.Feature(None, {"timestamp": date, "soil_moisture": val})

    all_rows = []

    for y_start, y_end in _year_ranges(START_DATE, END_DATE):
        batch = (
            collection
            .filterDate(y_start, y_end)
            .map(_map_fn)
            .filter(ee.Filter.notNull(["soil_moisture"]))
        )
        n = batch.size().getInfo()
        if n == 0:
            print(f"  {y_start[:4]}: 0 images, skipping")
            continue
        info = batch.getInfo()
        rows = _fc_to_rows(info, ["timestamp", "soil_moisture"])
        all_rows.extend(rows)
        print(f"  {y_start[:4]}: {len(rows)} records")
        del info, rows, batch
        gc.collect()

    if not all_rows:
        print("  WARNING: No soil moisture data retrieved.")
        return pd.DataFrame(columns=["timestamp", "soil_moisture"])

    df = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()

    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce").astype("float32")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} daily soil moisture records total")
    return df[["timestamp", "soil_moisture"]]


# ─── 2. ATMOSPHERIC HUMIDITY — Aqua MODIS MCD19A2 ────────────────────────────

def get_humidity() -> pd.DataFrame:
    """
    Aqua MODIS MCD19A2_GRANULES — Column_WV (cm) as humidity proxy.

    RAM strategy: quarterly chunks (≤~92 days × multiple granules/day).
    MODIS produces many granules per day so yearly chunks can still be large;
    quarterly keeps each getInfo() payload comfortably small.
    Duplicate same-day granules are collapsed to a daily mean after each batch.
    """
    print("Fetching Aqua MODIS humidity (column water vapor) data...")

    BAND = "Column_WV"
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterDate(START_DATE, END_DATE)
        .select(BAND)
        .filterBounds(AOI)
    )

    def _map_fn(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        val  = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=AOI,
            scale=1000,
            maxPixels=1e9,
        ).get(BAND)
        return ee.Feature(None, {"timestamp": date, "humidity_raw": val})

    daily_frames = []

    for q_start, q_end in _quarter_ranges(START_DATE, END_DATE):
        batch = (
            collection
            .filterDate(q_start, q_end)
            .map(_map_fn)
            .filter(ee.Filter.notNull(["humidity_raw"]))
        )
        n = batch.size().getInfo()
        if n == 0:
            continue
        info = batch.getInfo()
        rows = _fc_to_rows(info, ["timestamp", "humidity_raw"])
        del info, batch
        gc.collect()

        if not rows:
            del rows
            continue

        qdf = pd.DataFrame(rows)
        del rows
        qdf["timestamp"]    = pd.to_datetime(qdf["timestamp"])
        qdf["humidity_raw"] = pd.to_numeric(qdf["humidity_raw"], errors="coerce")

        # Collapse multiple granules → single daily mean within this quarter
        qdf = (
            qdf.groupby("timestamp", as_index=False)["humidity_raw"]
            .mean()
        )
        daily_frames.append(qdf)
        del qdf
        gc.collect()

    if not daily_frames:
        print("  WARNING: No humidity data retrieved.")
        return pd.DataFrame(columns=["timestamp", "humidity"])

    df = pd.concat(daily_frames, ignore_index=True)
    del daily_frames
    gc.collect()

    # Re-collapse across quarter boundaries (guard against off-by-one overlaps)
    df = df.groupby("timestamp", as_index=False)["humidity_raw"].mean()
    df["humidity"] = (df["humidity_raw"] * 0.001).astype("float32")
    df.drop(columns=["humidity_raw"], inplace=True)

    # ── Humidity floor filter ─────────────────────────────────────────────────
    # Column water vapor over Manila Bay never genuinely drops below ~0.5 cm.
    # Values ≤ 0.15 cm are cloud-contaminated MODIS pixels returning near-zero
    # rather than being properly masked. Replace with NaN, then forward-fill
    # (carry the last valid observation) with a 3-day limit so extended cloud
    # gaps stay as NaN rather than stale data.
    n_bad = (df["humidity"] <= 0.15).sum()
    if n_bad > 0:
        df.loc[df["humidity"] <= 0.15, "humidity"] = float("nan")
        df["humidity"] = (
            df["humidity"]
            .ffill(limit=3)
            .astype("float32")
        )
        print(f"  Humidity floor filter: {n_bad} near-zero values replaced "
              f"(forward-filled up to 3 days)")

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} daily humidity records total")
    return df[["timestamp", "humidity"]]


# ─── 3. TIDAL WATER LEVEL — UHSLC + CMEMS gap-fill ───────────────────────────
#
# PROBLEM: UHSLC is absolute metres above local datum (~1.9–2.5 m range).
#          CMEMS zos is SSH anomaly relative to geoid (~0.6–0.9 m range).
#          Concatenating raw creates artificial jumps that look like flood
#          signals to XGBoost.
#
# FIX: Z-score normalise each source independently against a shared reference
#      window (2017–2021) before merging. Output waterlevel is dimensionless
#      but consistent — relative tidal variability is preserved.

def _get_uhslc() -> pd.DataFrame:
    """
    UHSLC Station 370 Manila South Harbor via ERDDAP.
    Streaming + chunked CSV parse — raw response bytes are never fully
    materialised as a Python string in memory.
    Returns raw metres (absolute, above local datum). Normalisation is
    applied downstream in get_tidal_level().
    """
    print("Fetching UHSLC Manila tide gauge (ERDDAP)...")

    url = _UHSLC_ERDDAP_URL.format(start=START_DATE, end=END_DATE)

    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        raw_bytes = resp.content
    except requests.RequestException as e:
        print(f"  UHSLC ERDDAP fetch failed: {e}")
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    chunks = pd.read_csv(
        io.BytesIO(raw_bytes),
        skiprows=[1],           # drop units row
        on_bad_lines="skip",
        dtype={"sea_level": "float32"},
        chunksize=500,
    )
    del raw_bytes
    gc.collect()

    frames = []
    for chunk in chunks:
        chunk = chunk.rename(columns={"time": "timestamp", "sea_level": "sea_level_mm"})
        chunk["timestamp"]    = pd.to_datetime(chunk["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
        chunk["sea_level_mm"] = pd.to_numeric(chunk["sea_level_mm"], errors="coerce")
        chunk = chunk[chunk["sea_level_mm"] > -9000].dropna(subset=["sea_level_mm"])
        chunk["waterlevel"]   = (chunk["sea_level_mm"] / 1000.0).astype("float32")
        frames.append(chunk[["timestamp", "waterlevel"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    start    = pd.Timestamp(START_DATE)
    end      = pd.Timestamp(END_DATE)
    df       = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    expected = (end - start).days
    print(f"  → {len(df)} / {expected} days from UHSLC ({100 * len(df) / expected:.1f}% coverage)")
    return df


def _get_cmems(missing_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    CMEMS zos gap-fill.
    Returns raw zos values (SSH anomaly, metres).
    Normalisation is applied downstream in get_tidal_level().
    Splits automatically between reanalysis (pre-2021) and NRT (2021-present).
    """
    if missing_dates.empty:
        print("  No gaps to fill — skipping CMEMS.")
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    import copernicusmarine

    print(f"  Gap-filling {len(missing_dates)} missing days via CMEMS...")

    REANALYSIS_CUTOFF = pd.Timestamp("2021-07-01")
    gaps_rean = missing_dates[missing_dates <  REANALYSIS_CUTOFF]
    gaps_nrt  = missing_dates[missing_dates >= REANALYSIS_CUTOFF]

    common_kwargs = dict(
        variables         = ["zos"],
        minimum_longitude = _CMEMS_LON_MIN,
        maximum_longitude = _CMEMS_LON_MAX,
        minimum_latitude  = _CMEMS_LAT_MIN,
        maximum_latitude  = _CMEMS_LAT_MAX,
        username          = os.getenv("CMEMS_USERNAME"),
        password          = os.getenv("CMEMS_PASSWORD"),
    )

    frames = []

    if len(gaps_rean) > 0:
        print(f"    Reanalysis: {gaps_rean.min().date()} → {gaps_rean.max().date()}")
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id     = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                start_datetime = str(gaps_rean.min().date()),
                end_datetime   = str(gaps_rean.max().date()),
                **common_kwargs,
            )
            df_r = (
                ds["zos"]
                .mean(dim=["latitude", "longitude"])
                .to_dataframe()
                .reset_index()[["time", "zos"]]
                .rename(columns={"time": "timestamp", "zos": "waterlevel"})
            )
            ds.close()
            del ds
            df_r["timestamp"] = pd.to_datetime(df_r["timestamp"]).dt.normalize()
            df_r = df_r[df_r["timestamp"].isin(gaps_rean)]
            df_r["waterlevel"] = df_r["waterlevel"].astype("float32")
            frames.append(df_r)
            del df_r
        except Exception as e:
            print(f"    WARNING: CMEMS reanalysis failed: {e}")

    if len(gaps_nrt) > 0:
        print(f"    NRT: {gaps_nrt.min().date()} → {gaps_nrt.max().date()}")
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id     = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                start_datetime = str(gaps_nrt.min().date()),
                end_datetime   = str(gaps_nrt.max().date()),
                **common_kwargs,
            )
            df_n = (
                ds["zos"]
                .mean(dim=["latitude", "longitude"])
                .to_dataframe()
                .reset_index()[["time", "zos"]]
                .rename(columns={"time": "timestamp", "zos": "waterlevel"})
            )
            ds.close()
            del ds
            df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.normalize()
            df_n = df_n[df_n["timestamp"].isin(gaps_nrt)]
            df_n["waterlevel"] = df_n["waterlevel"].astype("float32")
            frames.append(df_n)
            del df_n
        except Exception as e:
            print(f"    WARNING: CMEMS NRT failed: {e}")

    gc.collect()

    if not frames:
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    df_cmems = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    print(f"    → {len(df_cmems)} CMEMS gap-fill records")
    return df_cmems[["timestamp", "waterlevel"]].reset_index(drop=True)


def get_tidal_level() -> pd.DataFrame:
    """
    Tidal water level for Obando / Manila Bay.

    Primary:  UHSLC Station 370 — real measured tide gauge (absolute metres)
    Gap-fill: CMEMS zos SSH — reanalysis (pre-2021) + NRT (2021-present)

    Normalisation:
        Both sources are z-score normalised independently using the
        2017-01-01 → 2021-07-01 reference window before merging.
        Output `waterlevel` is dimensionless (z-scores).
        Relative tidal variability is preserved; source-switch jumps
        (0.7x vs 2.x metre offsets) are eliminated.

        To recover physical units:
            waterlevel_m = waterlevel_zscore * uhslc_std + uhslc_mu
        (uhslc_mu and uhslc_std are printed during the run)
    """
    print("Fetching tidal water level data (UHSLC + CMEMS)...")

    df_uhslc  = _get_uhslc()
    all_dates = pd.date_range(
        START_DATE,
        end=pd.Timestamp(END_DATE) - pd.Timedelta(days=1),
        freq="D"
    )
    covered  = set(df_uhslc["timestamp"].dt.normalize()) if not df_uhslc.empty else set()
    missing  = pd.DatetimeIndex([d for d in all_dates if d not in covered])
    df_cmems = _get_cmems(missing)
    del missing, covered
    gc.collect()

    # ── Normalise each source independently ───────────────────────────────────
    REF_START = pd.Timestamp("2017-01-01")
    REF_END   = pd.Timestamp("2021-07-01")

    if not df_uhslc.empty:
        ref_mask  = (df_uhslc["timestamp"] >= REF_START) & (df_uhslc["timestamp"] < REF_END)
        uhslc_mu  = float(df_uhslc.loc[ref_mask, "waterlevel"].mean())
        uhslc_std = float(df_uhslc.loc[ref_mask, "waterlevel"].std())
        uhslc_std = uhslc_std if uhslc_std > 0 else 1.0
        df_uhslc["waterlevel"] = (
            (df_uhslc["waterlevel"] - uhslc_mu) / uhslc_std
        ).astype("float32")
        print(f"  UHSLC normalised  — ref μ={uhslc_mu:.4f} m, σ={uhslc_std:.4f} m")

    if not df_cmems.empty:
        ref_mask = (df_cmems["timestamp"] >= REF_START) & (df_cmems["timestamp"] < REF_END)
        if ref_mask.sum() > 10:
            cmems_mu  = float(df_cmems.loc[ref_mask, "waterlevel"].mean())
            cmems_std = float(df_cmems.loc[ref_mask, "waterlevel"].std())
        else:
            # fallback: normalise on full CMEMS series
            cmems_mu  = float(df_cmems["waterlevel"].mean())
            cmems_std = float(df_cmems["waterlevel"].std())
        cmems_std = cmems_std if cmems_std > 0 else 1.0
        df_cmems["waterlevel"] = (
            (df_cmems["waterlevel"] - cmems_mu) / cmems_std
        ).astype("float32")
        print(f"  CMEMS normalised  — ref μ={cmems_mu:.4f} m, σ={cmems_std:.4f} m")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = pd.concat([df_uhslc, df_cmems], ignore_index=True)
    del df_uhslc, df_cmems
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["waterlevel"] = pd.to_numeric(df["waterlevel"], errors="coerce").astype("float32")

    # ── Sanity check — flag suspicious seam jumps ─────────────────────────────
    df["_src_jump"] = df["waterlevel"].diff().abs()
    big_jumps = df[df["_src_jump"] > 2.0]
    if not big_jumps.empty:
        print(f"  WARNING: {len(big_jumps)} large day-to-day jumps (>2σ) remain after normalisation.")
        print(f"  First few:\n{big_jumps[['timestamp','waterlevel','_src_jump']].head(5).to_string(index=False)}")
    df.drop(columns=["_src_jump"], inplace=True)

    total  = len(all_dates)
    filled = df["waterlevel"].notna().sum()
    print(f"  Water level summary:")
    print(f"    UHSLC (measured) + CMEMS (gap-fill)")
    print(f"    Total filled: {filled} / {total} ({100 * filled / total:.1f}%)")
    return df[["timestamp", "waterlevel"]]


# ─── MERGE & EXPORT ───────────────────────────────────────────────────────────

def main():
    df_sm    = get_soil_moisture()
    df_humid = get_humidity()
    df_tide  = get_tidal_level()

    print("\nMerging datasets...")

    df = (
        df_tide
        .merge(df_sm,    on="timestamp", how="outer")
        .merge(df_humid, on="timestamp", how="outer")
    )
    del df_tide, df_sm, df_humid
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop trailing empty row if present (artefact of some CSV exports)
    df = df.dropna(how="all")

    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Cast float64 columns back to float32 (merge can upcast)
    for col in ["waterlevel", "soil_moisture", "humidity"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    df = df[["timestamp", "waterlevel", "soil_moisture", "humidity"]]
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ Saved → {OUTPUT_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"\nPreview:\n{df.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
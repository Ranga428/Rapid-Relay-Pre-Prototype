from pydsstools.heclib.dss import HecDss
from pydsstools.core import TimeSeriesContainer
import pandas as pd
import numpy as np

# ── 1. LOAD & INSPECT ──────────────────────────────────────────────────────────
df = pd.read_csv('HEC-HMS-Calibration_Data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print("=== RAW DATA INSPECTION ===")
print(f"Rows: {len(df)}")
print(f"Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"humidity   → min:{df['humidity'].min():.4f}  max:{df['humidity'].max():.4f}  mean:{df['humidity'].mean():.4f}")
print(f"soil_moist → min:{df['soil_moisture'].min():.4f}  max:{df['soil_moisture'].max():.4f}")
print(f"waterlevel → min:{df['waterlevel'].min():.4f}  max:{df['waterlevel'].max():.4f}")

# ── 2. HUMIDITY CONVERSION (corrected) ────────────────────────────────────────
hum_max  = df['humidity'].max()
hum_mean = df['humidity'].mean()

print(f"\nhumidity raw → min:{df['humidity'].min():.4f}  max:{hum_max:.4f}  mean:{hum_mean:.4f}")

if hum_max <= 1.05:
    # Fraction 0–1
    df['humidity_pct'] = (df['humidity'] * 100).clip(0, 100)
    print("Humidity detected as FRACTION → ×100")

elif hum_max <= 8.0 and hum_mean < 4.0:
    # Vapor pressure in kPa — Manila range 1.5–3.5 kPa
    # RH% = (e / es) × 100
    # es requires temperature. Without temp data, use Manila mean annual T = 28°C
    # es at 28°C = 0.6108 × exp(17.27×28 / (28+237.3)) = 3.782 kPa
    T_mean = 28.0
    es_kpa = 0.6108 * np.exp(17.27 * T_mean / (T_mean + 237.3))
    df['humidity_pct'] = ((df['humidity'] / es_kpa) * 100).clip(0, 100)
    print(f"Humidity detected as VAPOR PRESSURE kPa → converted using es={es_kpa:.3f} kPa at {T_mean}°C")

elif hum_max <= 100.0 and hum_mean > 35.0:
    # Already percent
    df['humidity_pct'] = df['humidity'].clip(0, 100)
    print("Humidity detected as already PERCENT → used as-is")

else:
    # Mixing ratio g/kg
    P_kpa  = 101.3
    T_mean = 28.0
    es_kpa = 0.6108 * np.exp(17.27 * T_mean / (T_mean + 237.3))
    e = (df['humidity'] * P_kpa) / (622 + df['humidity'])
    df['humidity_pct'] = ((e / es_kpa) * 100).clip(0, 100)
    print(f"Humidity detected as MIXING RATIO g/kg → converted to RH%")

print(f"humidity_pct → min:{df['humidity_pct'].min():.1f}  max:{df['humidity_pct'].max():.1f}  mean:{df['humidity_pct'].mean():.1f}")

if not (60 < df['humidity_pct'].mean() < 95):
    print("WARNING: humidity_pct mean still outside 60–95%. Manual inspection needed.")
    print("         Print df['humidity'].describe() and share the output.")

# ── 3. SOIL MOISTURE → PERCENT ─────────────────────────────────────────────────
# Your CSV is fraction 0–1. HEC-HMS SMA wants percent 0–100.
df['soil_moisture_pct'] = (df['soil_moisture'] * 100).clip(0, 100)

# ── 4. ZERO PRECIP PLACEHOLDER ─────────────────────────────────────────────────
df['zero_precip'] = 0.0

# ── 5. RESAMPLE TO STRICT 1-DAY GRID ──────────────────────────────────────────
df = (df.set_index('timestamp')
        .resample('1D')
        .interpolate(method='linear')
        .reset_index())

# Part D start date — must match the resampled start
part_d = df['timestamp'].iloc[0].strftime('%d%b%Y').upper()  # e.g. 01JAN2017
start_str = df['timestamp'].iloc[0].strftime('%d%b%Y %H:%M:%S').upper()

print(f"\nAfter resample: {len(df)} daily rows, starting {start_str}")

# ── 6. PATHNAME + COLUMN DEFINITIONS ──────────────────────────────────────────
# Format: /Part-A/Part-B/Part-C/Part-D/Part-E/Part-F/
# Part D = block start date (required for HEC-HMS 4.13 lookup)
# Part C names must exactly match HEC-HMS gage type expectations

sensors = [
    # (dataframe_col,      Part-C,              units,     dss_type,  description)
    ('waterlevel',         'STAGE',             'M',       'INST-VAL', 'Water level gage'),
    ('soil_moisture_pct',  'SOIL MOISTURE',     'PERCENT', 'INST-VAL', 'Soil moisture gage'),
    ('humidity_pct',       'RELATIVE HUMIDITY', 'PERCENT', 'INST-VAL', 'Humidity gage'),
    ('zero_precip',        'PRECIP-INC',        'MM',      'PER-CUM',  'Precipitation placeholder'),
]

# ── 7. WRITE TO DSS ────────────────────────────────────────────────────────────
dss_file = 'sensor_data.dss'

with HecDss.Open(dss_file) as dss:
    for col, parameter, units, data_type, desc in sensors:
        pathname = f'/YOUR-RIVER/STATION1/{parameter}/{part_d}/1DAY/OBS/'

        tsc = TimeSeriesContainer()
        tsc.pathname     = pathname
        tsc.startDateTime = start_str
        tsc.numberValues = len(df)
        tsc.units        = units
        tsc.type         = data_type
        tsc.interval     = 1440   # minutes — 1 day
        tsc.values       = df[col].values.astype(float)

        dss.put(tsc)
        print(f"Written: {pathname}")

# ── 8. VERIFY — READ BACK AND CONFIRM ─────────────────────────────────────────
print("\n=== VERIFICATION (read-back) ===")
with HecDss.Open(dss_file) as dss:
    for col, parameter, units, data_type, desc in sensors:
        pathname = f'/YOUR-RIVER/STATION1/{parameter}/{part_d}/1DAY/OBS/'
        try:
            ts = dss.read_ts(pathname)
            vals = np.array(ts.values)
            # Filter out HEC-DSS no-data sentinel (-3.4028235e+38)
            valid = vals[vals > -1e+30]
            print(f"  {parameter:<22} → {len(valid):>4} values  "
                  f"min={valid.min():.3f}  max={valid.max():.3f}  "
                  f"mean={valid.mean():.3f}  nulls={len(vals)-len(valid)}")
        except Exception as e:
            print(f"  {parameter:<22} → READ FAILED: {e}")

print('\nDone. sensor_data.dss is ready for HEC-HMS 4.13.')
print(f'Part D in all pathnames: {part_d}')
print('Use this exact Part D when pointing gages in Time-Series Data Manager.')
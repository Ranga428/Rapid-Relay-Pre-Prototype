from datetime import datetime
from pathlib import Path
import csv


def _format_eo_from_values(soil_val, flood_val, trend_val):
    try:
        soil_pct = f"{round(float(soil_val) * 100, 1)}%" if soil_val is not None and soil_val != "" else "N/A"
    except Exception:
        soil_pct = "N/A"
    try:
        flood_pct = f"{round(float(flood_val) * 100, 1)}%" if flood_val is not None and flood_val != "" else "N/A"
    except Exception:
        flood_pct = "N/A"

    if trend_val is None or trend_val == "":
        trend_str = "N/A"
    else:
        trend_map = {1: "increasing", 0: "stable", -1: "decreasing"}
        try:
            trend_str = trend_map.get(int(float(trend_val)), str(trend_val))
        except Exception:
            trend_str = str(trend_val)

    return soil_pct, flood_pct, trend_str


def notify(csv_path=None):
    """Show all rows from the merged events CSV using the project's message format.

    This function only reads `logs/events.csv` (or `csv_path` if provided) and
    prints one formatted line per row. It does not perform any other IO or
    computation.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "logs" / "events.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"No events CSV found at {csv_path}")
        return

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            level = (r.get('warning_level') or "GREEN").upper()
            if level not in ("RED", "YELLOW", "GREEN"):
                level = "GREEN"

            risk = r.get('risk')

            sensor_rec = {
                'timestamp': r.get('timestamp'),
                'water_level': r.get('water_level') or "",
                'rainfall': r.get('rainfall') or "",
                'humidity': r.get('humidity') or "",
            }
            soil_val = r.get('soil_saturation') or None
            flood_val = r.get('flood_extent') or None
            trend_val = r.get('wetness_trend') or None

            soil_pct, flood_pct, trend_str = _format_eo_from_values(soil_val, flood_val, trend_val)

            prefix = {
                'RED': '[RED ALERT]',
                'YELLOW': '[YELLOW WARNING]',
                'GREEN': '[GREEN SAFETY]',
            }[level]

            timestamp = sensor_rec.get('timestamp')

            msg = (
                f"{prefix} {timestamp} | "
                f"Level: {level} | "
                f"Water Level: {sensor_rec.get('water_level')} m | "
                f"Rainfall: {sensor_rec.get('rainfall')} mm | "
                f"Humidity: {sensor_rec.get('humidity')} % | "
                f"Soil: {soil_pct} | Flood extent: {flood_pct} | Wetness trend: {trend_str}"
            )

            if risk is not None and risk != "":
                try:
                    msg = f"{msg} | Risk: {round(float(risk), 3)}"
                except Exception:
                    msg = f"{msg} | Risk: {risk}"

            print(msg)

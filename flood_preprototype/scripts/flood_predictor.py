"""
flood_predictor.py
==================
Real-time flood prediction class wrapper.

Loads a trained XGBoost model and exposes a clean API for:
  - Single-timestamp predictions
  - Batch predictions over a DataFrame
  - Alert generation with configurable thresholds

Usage example
-------------
    from flood_predictor import FloodPredictor
    from feature_engineering import build_features

    predictor = FloodPredictor("model/flood_xgb_model.pkl")

    # Real-time: pass last 24h of raw sensor data
    features = build_features(raw_sensor_df)
    result   = predictor.predict_latest(features)
    print(result)

    # Batch: run over a full feature DataFrame
    results = predictor.predict_batch(features_df)
"""

import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

@dataclass
class AlertLevel:
    name:  str
    color: str
    emoji: str


ALERT_LEVELS = {
    "CLEAR":   AlertLevel("CLEAR",   "green",  "🟢"),
    "WATCH":   AlertLevel("WATCH",   "yellow", "🟡"),
    "WARNING": AlertLevel("WARNING", "orange", "🟠"),
    "DANGER":  AlertLevel("DANGER",  "red",    "🔴"),
}


@dataclass
class PredictionResult:
    timestamp:       pd.Timestamp
    flood_probability: float
    flood_predicted:  int
    alert_level:     str
    alert_emoji:     str
    features_used:   dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.alert_emoji} [{self.timestamp}]  "
            f"Flood probability: {self.flood_probability:.1%}  "
            f"→  {self.alert_level}"
        )


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class FloodPredictor:
    """
    Real-time flood prediction wrapper around a trained XGBoost model.

    Parameters
    ----------
    model_path : str
        Path to a joblib-saved XGBClassifier.
    watch_threshold   : float
        Probability above which a WATCH alert is issued.
    warning_threshold : float
        Probability above which a WARNING alert is issued.
    danger_threshold  : float
        Probability above which a DANGER alert is issued.
    feature_columns : list[str] | None
        Ordered list of feature column names the model expects.
        If None, defaults to the 14 standard features.
    """

    DEFAULT_FEATURES = [
        "max_water_level_6h",
        "max_water_level_24h",
        "water_level_slope_3h",
        "water_level_slope_6h",
        "water_level_std_24h",
        "rainfall_sum_1h",
        "rainfall_sum_6h",
        "rainfall_sum_24h",
        "rainfall_max_intensity",
        "soil_moisture_mean_24h",
        "soil_moisture_trend_48h",
        "tidal_height_current",
        "tidal_height_max_24h",
        "tidal_phase",
    ]

    def __init__(
        self,
        model_path:        str,
        watch_threshold:   float = 0.40,
        warning_threshold: float = 0.60,
        danger_threshold:  float = 0.80,
        feature_columns:   Optional[list] = None,
    ):
        self.model             = self._load_model(model_path)
        self.watch_threshold   = watch_threshold
        self.warning_threshold = warning_threshold
        self.danger_threshold  = danger_threshold
        self.feature_columns   = feature_columns or self.DEFAULT_FEATURES

        logger.info(
            f"FloodPredictor ready | Thresholds — "
            f"WATCH:{watch_threshold:.0%}  "
            f"WARNING:{warning_threshold:.0%}  "
            f"DANGER:{danger_threshold:.0%}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_latest(self, features_df: pd.DataFrame) -> PredictionResult:
        """
        Run inference on the most recent row of a feature DataFrame.

        Args:
            features_df : DataFrame with datetime index and feature columns.

        Returns:
            PredictionResult for the latest timestamp.
        """
        row = features_df.iloc[[-1]]
        return self._predict_row(row)

    def predict_at(
        self,
        features_df: pd.DataFrame,
        timestamp:   pd.Timestamp,
    ) -> PredictionResult:
        """
        Run inference for a specific timestamp.

        Args:
            features_df : Feature DataFrame with datetime index.
            timestamp   : Target timestamp (exact match).

        Returns:
            PredictionResult for that timestamp.
        """
        row = features_df.loc[[timestamp]]
        return self._predict_row(row)

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference over a full feature DataFrame.

        Args:
            features_df : Feature DataFrame with datetime index.

        Returns:
            DataFrame with columns:
                flood_probability, flood_predicted, alert_level
        """
        X = self._validate_features(features_df)
        probs  = self.model.predict_proba(X)[:, 1]
        preds  = (probs >= self.warning_threshold).astype(int)
        alerts = [self._classify_alert(p) for p in probs]

        results = pd.DataFrame({
            "flood_probability": probs,
            "flood_predicted":   preds,
            "alert_level":       alerts,
        }, index=features_df.index)

        return results

    def stream_predict(self, sensor_stream, feature_builder_fn, interval_s: int = 900):
        """
        Continuous prediction loop for a live sensor stream.

        Args:
            sensor_stream     : Iterable that yields raw sensor DataFrames
                                (each containing the latest 24h+ of data).
            feature_builder_fn: Callable — takes raw sensor DataFrame,
                                returns engineered feature DataFrame.
                                Typically `feature_engineering.build_features`.
            interval_s        : Sleep interval between predictions (seconds).
                                Default 900 = 15 minutes.

        Yields:
            PredictionResult for each sensor batch.
        """
        import time
        logger.info("Starting real-time prediction loop…")
        for raw_batch in sensor_stream:
            try:
                features = feature_builder_fn(raw_batch)
                result   = self.predict_latest(features)
                logger.info(str(result))
                yield result
                time.sleep(interval_s)
            except Exception as exc:
                logger.error(f"Prediction error: {exc}")
                time.sleep(interval_s)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, path: str):
        logger.info(f"Loading model from {path}")
        model = joblib.load(path)
        return model

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return df[self.feature_columns]

    def _predict_row(self, row: pd.DataFrame) -> PredictionResult:
        X    = self._validate_features(row)
        prob = float(self.model.predict_proba(X)[0][1])
        pred = int(prob >= self.warning_threshold)
        al   = self._classify_alert(prob)

        return PredictionResult(
            timestamp          = row.index[0],
            flood_probability  = prob,
            flood_predicted    = pred,
            alert_level        = al,
            alert_emoji        = ALERT_LEVELS[al].emoji,
            features_used      = row.iloc[0].to_dict(),
        )

    def _classify_alert(self, prob: float) -> str:
        if prob >= self.danger_threshold:
            return "DANGER"
        elif prob >= self.warning_threshold:
            return "WARNING"
        elif prob >= self.watch_threshold:
            return "WATCH"
        else:
            return "CLEAR"


# ---------------------------------------------------------------------------
# Quick demo (no real model needed — uses a mock)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import xgboost as xgb
    import tempfile, os

    print("Running FloodPredictor smoke-test with mock model…\n")

    # Create a tiny dummy model
    n_features = 14
    rng = np.random.default_rng(0)
    X_dummy = rng.random((100, n_features))
    y_dummy = rng.integers(0, 2, 100)

    mock_model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric="logloss")
    mock_model.fit(X_dummy, y_dummy)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    joblib.dump(mock_model, tmp_path)

    # Build a mock feature DataFrame
    idx = pd.date_range("2024-06-01", periods=20, freq="15min")
    feat_df = pd.DataFrame(rng.random((20, n_features)), index=idx,
                           columns=FloodPredictor.DEFAULT_FEATURES)

    predictor = FloodPredictor(tmp_path)

    # Latest prediction
    result = predictor.predict_latest(feat_df)
    print(result)

    # Batch
    batch = predictor.predict_batch(feat_df)
    print("\nBatch results:")
    print(batch.head())

    os.unlink(tmp_path)
    print("\nSmoke-test passed ✅")

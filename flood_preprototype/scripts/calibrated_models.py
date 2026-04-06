"""
calibrated_models.py
====================
Shared calibrated wrapper classes for the Flood Early Warning System.

WHY THIS FILE EXISTS
--------------------
joblib/pickle stores class references by (module, qualname). When a class
is defined inside a training script that runs as __main__, the pkl records
it as '__main__.ClassName'. When Start.py later imports a predict script
and calls joblib.load(), __main__ is Start.py — which has no such class —
so the load crashes with:

    AttributeError: Can't get attribute 'CalibratedRF' on <module '__main__'>

The fix: define all wrapper classes here, in a stable importable module.
The pkl will then record them as 'calibrated_models.CalibratedRF' and
'calibrated_models.CalibratedLGBM', which resolves correctly regardless
of which script is __main__.

WHICH MODELS USE THIS
---------------------
    CalibratedRF   — RF_train_flood_model.py   → flood_rf_sensor.pkl
    CalibratedLGBM — LGBM_train_flood_model.py → flood_lgbm_sensor.pkl
    XGBClassifier  — XGB_train_flood_model.py  → flood_xgb_sensor.pkl
                     (XGB saves its native classifier directly, no wrapper needed)

USAGE
-----
In every training script and every predict script that loads these pkls:

    from calibrated_models import CalibratedRF, CalibratedLGBM

Remove the inline class definitions from:
    RF_train_flood_model.py
    LGBM_train_flood_model.py
    RF_Predict.py
    LGBM_Predict.py  (if it defines the class inline)

After updating the training scripts, retrain both models so the new pkls
record the stable 'calibrated_models.*' class reference.
"""

import numpy as np


class CalibratedRF:
    """
    Isotonic-calibrated wrapper for RandomForestClassifier.

    Wraps a fitted base RF and a fitted IsotonicRegression calibrator.
    predict_proba() passes raw RF probabilities through the calibrator
    before returning them.

    The calibrator is fitted on a held-out calibration fold during training
    (last 20% of the training period) — its learned mapping is stored inside
    this instance and serialised into the pkl alongside the base model.
    """

    def __init__(self, base, calibrator):
        self.base       = base        # fitted RandomForestClassifier
        self.calibrator = calibrator  # fitted IsotonicRegression
        self.estimator  = base        # sklearn compatibility alias

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self):
        return self.base.classes_

    @property
    def n_features_in_(self):
        return self.base.n_features_in_

    def __repr__(self):
        return (f"CalibratedRF(base={self.base.__class__.__name__}, "
                f"calibrator={self.calibrator.__class__.__name__})")


class CalibratedLGBM:
    """
    Isotonic-calibrated wrapper for LGBMClassifier.

    Identical structure to CalibratedRF — wraps a fitted LGBMClassifier
    and a fitted IsotonicRegression calibrator trained on a held-out
    calibration fold (last 20% of the training period).
    """

    def __init__(self, base, calibrator):
        self.base       = base        # fitted LGBMClassifier
        self.calibrator = calibrator  # fitted IsotonicRegression
        self.estimator  = base        # sklearn compatibility alias

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self):
        return self.base.classes_

    @property
    def n_features_in_(self):
        return self.base.n_features_in_

    def __repr__(self):
        return (f"CalibratedLGBM(base={self.base.__class__.__name__}, "
                f"calibrator={self.calibrator.__class__.__name__})")
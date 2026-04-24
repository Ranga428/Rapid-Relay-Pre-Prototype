#!/usr/bin/env python
import pickle
from pathlib import Path

model_path = Path('../model/flood_xgb_sensor.pkl')
if model_path.exists():
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    print('Model Thresholds Loaded from artifact:')
    watch_t = artifact.get("watch_threshold")
    warn_t = artifact.get("warning_threshold")
    danger_t = artifact.get("danger_threshold")
    print(f'  WATCH   : {watch_t}')
    print(f'  WARNING : {warn_t}')
    print(f'  DANGER  : {danger_t}')
    if watch_t and warn_t:
        print(f'\n  WATCH → WARNING gap: {warn_t - watch_t:.3f}')
    if warn_t and danger_t:
        print(f'  WARNING → DANGER gap: {danger_t - warn_t:.3f}')
else:
    print('Model file not found')

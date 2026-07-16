# Level-2

This folder contains the Python implementation of the open-field CoMPASS Level-2 workflow.

## Level-2 datastreams

The Level-2 model uses three open-field datastreams:

1. `KDE` — spatial occupancy density calculated from x/y position using a two-dimensional Gaussian KDE with Silverman's bandwidth rule.
2. `dist_from_wall` — radial distance from the animal to the circular arena wall.
3. `abs_movement_angle_to_wall_rad` — unsigned angle between the animal's next movement vector and the outward wall-normal vector.

The Level-1 state is also passed to Level-2 as a categorical model input:

4. `HMM_State` — Level-1 Surveillance/Ambulatory state.

The standard Level-2 feature list is therefore:

```python
features = [
    "KDE",
    "dist_from_wall",
    "abs_movement_angle_to_wall_rad",
    "HMM_State",
]
```

## Files

- `datastreams.py` — calculates KDE, wall distance, and movement angle relative to the wall normal.
- `preprocessing.py` — row ordering, feature validation, standardization, categorical one-hot encoding, session splitting, and forward-chaining splits.
- `model.py` — BGMM initialization, state-specific GMM initialization, GMM-HMM fitting, restart selection, AIC calculation, and nested tuning.
- `pipeline.py` — outer session-held-out evaluation, final all-session fitting, and per-mouse/dictionary entry points.
- `__init__.py` — package-level imports.

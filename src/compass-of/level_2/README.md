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


### `level2_datastreams.py`

This file **creates the actual behavioral datastreams**. It does not standardize features or fit a model.

Generated continuous variables:

```python
LEVEL2_CONTINUOUS_FEATURES = [
    "KDE",
    "dist_from_wall",
    "abs_movement_angle_to_wall_rad",
]
```

Carried forward from Level-1:

```python
LEVEL2_CATEGORICAL_FEATURES = ["HMM_State"]
```

Final Level-2 model inputs:

```python
LEVEL2_MODEL_FEATURES = [
    "KDE",
    "dist_from_wall",
    "abs_movement_angle_to_wall_rad",
    "HMM_State",
]
```

Calculations:

1. Low-likelihood belly and sternum DLC coordinates are removed and interpolated within each mouse/session.
2. Body-center position is the belly/sternum midpoint.
3. `KDE` is a 2D Silverman occupancy KDE evaluated at each body-center coordinate, separately per mouse.
4. `dist_from_wall = arena_radius - distance_from_center`.
5. The body vector `belly → sternum` is compared with the outward wall-normal vector `arena center → body center`.
6. `movement_angle_to_wall_rad` is the signed angle in `[-pi, pi]`; `abs_movement_angle_to_wall_rad` is in `[0, pi]`.
7. Existing Level-1 `HMM_State` values are validated and retained; they are not recomputed.

### `level2_preprocessing.py`

This is the separate model-preprocessing file from the supplied Level-2 code. It handles:

- row/session ordering
- feature validation
- training-only continuous-feature standardization
- categorical one-hot encoding of `HMM_State`
- invalid-row filtering
- sequence assembly
- forward-chaining splits

### `level2_model.py`

Contains the supplied BGMM initialization, state-specific GMM initialization, diagonal GMM-HMM, restart fitting, AIC calculation, and nested hyperparameter tuning.

### `level2_pipeline.py`

Contains outer session-held-out evaluation, final all-session fitting, one-mouse execution, and dictionary-of-mice execution.


"""Create the datastreams used by the CoMPASS Level-2 open-field model.

This module only constructs the actual Level-2 input variables:

- ``KDE``: two-dimensional occupancy density from body-center coordinates.
- ``dist_from_wall``: radial distance from the body center to the circular wall.
- ``movement_angle_to_wall_rad``: signed belly-to-sternum body angle relative
  to the outward wall-normal vector.
- ``abs_movement_angle_to_wall_rad``: absolute value of that signed angle.
- ``HMM_State``: carried forward from the fitted Level-1 model and treated as
  a categorical feature later by ``level2_preprocessing.py``.

No standardization, one-hot encoding, train/test splitting, or model fitting is
performed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

__all__ = [
    "LEVEL2_CONTINUOUS_FEATURES",
    "LEVEL2_CATEGORICAL_FEATURES",
    "LEVEL2_MODEL_FEATURES",
    "clean_dlc_coordinates",
    "calculate_body_center",
    "estimate_circle_from_extremes",
    "calculate_kde",
    "calculate_distance_from_wall",
    "calculate_movement_angle_to_wall",
    "create_level2_datastreams",
    "create_level2_datastreams_dict",
]

LEVEL2_CONTINUOUS_FEATURES = [
    "KDE",
    "dist_from_wall",
    "abs_movement_angle_to_wall_rad",
]

LEVEL2_CATEGORICAL_FEATURES = ["HMM_State"]

LEVEL2_MODEL_FEATURES = (
    LEVEL2_CONTINUOUS_FEATURES + LEVEL2_CATEGORICAL_FEATURES
)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _present_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _circle_values(circle: Mapping[str, Any]) -> tuple[float, float, float]:
    """Read a circle dictionary using common center/radius key names."""
    cx = circle.get("cx", circle.get("center_x", circle.get("arena_center_x")))
    cy = circle.get("cy", circle.get("center_y", circle.get("arena_center_y")))
    radius = circle.get("r", circle.get("radius", circle.get("arena_radius")))

    if cx is None or cy is None or radius is None:
        raise ValueError(
            "Circle information must contain cx/cy/r, center_x/center_y/radius, "
            "or arena_center_x/arena_center_y/arena_radius."
        )

    cx = float(cx)
    cy = float(cy)
    radius = float(radius)

    if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        raise ValueError("Circle center and radius must be finite.")
    if radius <= 0:
        raise ValueError("Circle radius must be greater than zero.")

    return cx, cy, radius


def _circle_for_mouse(
    circle_info: Mapping[Any, Any],
    mouse_id: Any,
) -> tuple[float, float, float]:
    """Resolve either one shared circle or a dictionary keyed by mouse."""
    shared_circle_keys = {
        "cx",
        "cy",
        "r",
        "center_x",
        "center_y",
        "radius",
        "arena_center_x",
        "arena_center_y",
        "arena_radius",
    }

    if any(key in circle_info for key in shared_circle_keys):
        return _circle_values(circle_info)

    if mouse_id in circle_info:
        return _circle_values(circle_info[mouse_id])
    if str(mouse_id) in circle_info:
        return _circle_values(circle_info[str(mouse_id)])

    raise ValueError(f"No circle information found for mouse {mouse_id!r}.")


def clean_dlc_coordinates(
    df: pd.DataFrame,
    belly_x_col: str = "belly_x",
    belly_y_col: str = "belly_y",
    sternum_x_col: str = "sternum_x",
    sternum_y_col: str = "sternum_y",
    belly_likelihood_col: str | None = "belly_likelihood",
    sternum_likelihood_col: str | None = "sternum_likelihood",
    likelihood_threshold: float = 0.8,
    mouse_col: str = "MouseID",
    session_col: str = "Session",
    order_col: str = "S_no",
) -> pd.DataFrame:
    """Filter low-likelihood DLC points and interpolate within each session.

    Row order is restored before the DataFrame is returned.
    """
    coordinate_columns = [
        belly_x_col,
        belly_y_col,
        sternum_x_col,
        sternum_y_col,
    ]
    _require_columns(df, coordinate_columns)

    out = df.copy()
    row_order_col = "__level2_original_row_order__"
    if row_order_col in out.columns:
        raise ValueError(f"Reserved temporary column already exists: {row_order_col}")
    out[row_order_col] = np.arange(len(out), dtype=int)

    sort_columns = _present_columns(out, [mouse_col, session_col, order_col])
    if sort_columns:
        out = out.sort_values(sort_columns, kind="mergesort").copy()

    for column in coordinate_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if belly_likelihood_col is not None and belly_likelihood_col in out.columns:
        likelihood = pd.to_numeric(out[belly_likelihood_col], errors="coerce")
        out.loc[
            likelihood < likelihood_threshold,
            [belly_x_col, belly_y_col],
        ] = np.nan

    if sternum_likelihood_col is not None and sternum_likelihood_col in out.columns:
        likelihood = pd.to_numeric(out[sternum_likelihood_col], errors="coerce")
        out.loc[
            likelihood < likelihood_threshold,
            [sternum_x_col, sternum_y_col],
        ] = np.nan

    group_columns = _present_columns(out, [mouse_col, session_col])
    if group_columns:
        out[coordinate_columns] = (
            out.groupby(group_columns, sort=False, dropna=False)[coordinate_columns]
            .transform(
                lambda values: values.interpolate(
                    method="linear",
                    limit_direction="both",
                )
            )
        )
    else:
        out[coordinate_columns] = out[coordinate_columns].interpolate(
            method="linear",
            limit_direction="both",
        )

    out = out.sort_values(row_order_col, kind="mergesort")
    out = out.drop(columns=row_order_col).reset_index(drop=True)
    return out


def calculate_body_center(
    df: pd.DataFrame,
    belly_x_col: str = "belly_x",
    belly_y_col: str = "belly_y",
    sternum_x_col: str = "sternum_x",
    sternum_y_col: str = "sternum_y",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Set body-center x/y to the belly-sternum midpoint."""
    required = [belly_x_col, belly_y_col, sternum_x_col, sternum_y_col]
    _require_columns(df, required)

    out = df.copy()
    belly_x = pd.to_numeric(out[belly_x_col], errors="coerce")
    belly_y = pd.to_numeric(out[belly_y_col], errors="coerce")
    sternum_x = pd.to_numeric(out[sternum_x_col], errors="coerce")
    sternum_y = pd.to_numeric(out[sternum_y_col], errors="coerce")

    out[x_col] = (belly_x + sternum_x) / 2.0
    out[y_col] = (belly_y + sternum_y) / 2.0
    return out


def estimate_circle_from_extremes(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
) -> dict[str, float]:
    """Estimate a circular arena from the tracked coordinate extremes.

    This is a fallback. Supplying measured ``circle_info`` is preferable.
    """
    _require_columns(df, [x_col, y_col])
    xy = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(xy) < 4:
        raise ValueError("At least four finite x/y rows are required.")

    x_min = float(xy[x_col].min())
    x_max = float(xy[x_col].max())
    y_min = float(xy[y_col].min())
    y_max = float(xy[y_col].max())

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    extreme_indices = [
        xy[x_col].idxmin(),
        xy[x_col].idxmax(),
        xy[y_col].idxmin(),
        xy[y_col].idxmax(),
    ]
    extreme_points = xy.loc[extreme_indices, [x_col, y_col]].to_numpy(float)
    radius = float(
        np.mean(
            np.sqrt(
                (extreme_points[:, 0] - cx) ** 2
                + (extreme_points[:, 1] - cy) ** 2
            )
        )
    )

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Could not estimate a valid circular-arena radius.")

    return {"cx": cx, "cy": cy, "r": radius}


def calculate_kde(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    mouse_col: str = "MouseID",
    output_col: str = "KDE",
    bw_method: str | float = "silverman",
    normalize_0_to_1: bool = False,
) -> pd.DataFrame:
    """Evaluate a two-dimensional occupancy KDE at every tracked position.

    KDE is fit independently for each mouse when ``mouse_col`` is present.
    Raw Silverman KDE values are returned by default because Level-2 later
    standardizes continuous features using training data only.
    """
    _require_columns(df, [x_col, y_col])
    out = df.copy()
    out[output_col] = np.nan

    groups = (
        out.groupby(mouse_col, sort=False, dropna=False)
        if mouse_col in out.columns
        else [(None, out)]
    )

    for _, group in groups:
        x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        density = np.full(len(group), np.nan, dtype=float)

        if valid.sum() >= 3:
            positions = np.vstack([x[valid], y[valid]])
            try:
                kde = gaussian_kde(positions, bw_method=bw_method)
                fitted_density = kde(positions)
            except np.linalg.LinAlgError:
                # Deterministic, scale-aware jitter for degenerate trajectories.
                scale_x = max(float(np.nanstd(positions[0])), 1.0)
                scale_y = max(float(np.nanstd(positions[1])), 1.0)
                jitter = np.linspace(-1.0, 1.0, valid.sum())
                jittered = positions.copy()
                jittered[0] += jitter * scale_x * 1e-9
                jittered[1] -= jitter * scale_y * 1e-9
                kde = gaussian_kde(jittered, bw_method=bw_method)
                fitted_density = kde(jittered)

            if normalize_0_to_1:
                minimum = float(np.min(fitted_density))
                maximum = float(np.max(fitted_density))
                if np.isclose(minimum, maximum):
                    fitted_density = np.zeros_like(fitted_density)
                else:
                    fitted_density = (
                        fitted_density - minimum
                    ) / (maximum - minimum)

            density[valid] = fitted_density

        out.loc[group.index, output_col] = density

    return out


def calculate_distance_from_wall(
    df: pd.DataFrame,
    circle_info: Mapping[Any, Any],
    x_col: str = "x",
    y_col: str = "y",
    mouse_col: str = "MouseID",
    output_col: str = "dist_from_wall",
) -> pd.DataFrame:
    """Calculate ``radius - distance_from_arena_center``."""
    _require_columns(df, [x_col, y_col])
    out = df.copy()

    out["dist_from_center"] = np.nan
    out[output_col] = np.nan
    out["arena_center_x"] = np.nan
    out["arena_center_y"] = np.nan
    out["arena_radius"] = np.nan

    groups = (
        out.groupby(mouse_col, sort=False, dropna=False)
        if mouse_col in out.columns
        else [(None, out)]
    )

    for mouse_id, group in groups:
        cx, cy, radius = _circle_for_mouse(circle_info, mouse_id)
        x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(float)
        dist_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        out.loc[group.index, "dist_from_center"] = dist_center
        out.loc[group.index, output_col] = radius - dist_center
        out.loc[group.index, "arena_center_x"] = cx
        out.loc[group.index, "arena_center_y"] = cy
        out.loc[group.index, "arena_radius"] = radius

    return out


def calculate_movement_angle_to_wall(
    df: pd.DataFrame,
    belly_x_col: str = "belly_x",
    belly_y_col: str = "belly_y",
    sternum_x_col: str = "sternum_x",
    sternum_y_col: str = "sternum_y",
    x_col: str = "x",
    y_col: str = "y",
    output_col: str = "movement_angle_to_wall_rad",
    abs_output_col: str = "abs_movement_angle_to_wall_rad",
) -> pd.DataFrame:
    """Compare belly→sternum direction with the outward wall normal.

    ``movement_angle_to_wall_rad`` is signed and lies in [-pi, pi].
    ``abs_movement_angle_to_wall_rad`` lies in [0, pi].
    """
    required = [
        belly_x_col,
        belly_y_col,
        sternum_x_col,
        sternum_y_col,
        x_col,
        y_col,
        "arena_center_x",
        "arena_center_y",
    ]
    _require_columns(df, required)
    out = df.copy()

    belly_x = pd.to_numeric(out[belly_x_col], errors="coerce").to_numpy(float)
    belly_y = pd.to_numeric(out[belly_y_col], errors="coerce").to_numpy(float)
    sternum_x = pd.to_numeric(out[sternum_x_col], errors="coerce").to_numpy(float)
    sternum_y = pd.to_numeric(out[sternum_y_col], errors="coerce").to_numpy(float)
    x = pd.to_numeric(out[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(float)
    cx = pd.to_numeric(out["arena_center_x"], errors="coerce").to_numpy(float)
    cy = pd.to_numeric(out["arena_center_y"], errors="coerce").to_numpy(float)

    body_x = sternum_x - belly_x
    body_y = sternum_y - belly_y
    normal_x = x - cx
    normal_y = y - cy

    body_norm = np.hypot(body_x, body_y)
    normal_norm = np.hypot(normal_x, normal_y)
    denominator = body_norm * normal_norm

    valid = (
        np.isfinite(body_x)
        & np.isfinite(body_y)
        & np.isfinite(normal_x)
        & np.isfinite(normal_y)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    cosine = np.full(len(out), np.nan, dtype=float)
    sine = np.full(len(out), np.nan, dtype=float)
    cosine[valid] = (
        body_x[valid] * normal_x[valid]
        + body_y[valid] * normal_y[valid]
    ) / denominator[valid]
    sine[valid] = (
        body_x[valid] * normal_y[valid]
        - body_y[valid] * normal_x[valid]
    ) / denominator[valid]
    cosine[valid] = np.clip(cosine[valid], -1.0, 1.0)
    sine[valid] = np.clip(sine[valid], -1.0, 1.0)

    signed_angle = np.full(len(out), np.nan, dtype=float)
    signed_angle[valid] = np.arctan2(sine[valid], cosine[valid])

    out["cos_movement_angle_to_wall"] = cosine
    out[output_col] = signed_angle
    out[abs_output_col] = np.abs(signed_angle)
    return out


def create_level2_datastreams(
    df: pd.DataFrame,
    circle_info: Mapping[Any, Any] | None = None,
    mouse_col: str = "MouseID",
    session_col: str = "Session",
    order_col: str = "S_no",
    hmm_state_col: str = "HMM_State",
    belly_x_col: str = "belly_x",
    belly_y_col: str = "belly_y",
    sternum_x_col: str = "sternum_x",
    sternum_y_col: str = "sternum_y",
    belly_likelihood_col: str | None = "belly_likelihood",
    sternum_likelihood_col: str | None = "sternum_likelihood",
    likelihood_threshold: float = 0.8,
    x_col: str = "x",
    y_col: str = "y",
    calculate_xy_from_bodyparts: bool = True,
    estimate_circle_if_missing: bool = True,
    normalize_kde_0_to_1: bool = False,
) -> pd.DataFrame:
    """Create KDE, wall distance, wall angle, and retain Level-1 HMM state."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty.")
    if hmm_state_col not in df.columns:
        raise ValueError(
            f"Missing {hmm_state_col!r}. Run Level-1 first and append its "
            "decoded HMM_State before constructing Level-2 datastreams."
        )

    out = clean_dlc_coordinates(
        df=df,
        belly_x_col=belly_x_col,
        belly_y_col=belly_y_col,
        sternum_x_col=sternum_x_col,
        sternum_y_col=sternum_y_col,
        belly_likelihood_col=belly_likelihood_col,
        sternum_likelihood_col=sternum_likelihood_col,
        likelihood_threshold=likelihood_threshold,
        mouse_col=mouse_col,
        session_col=session_col,
        order_col=order_col,
    )

    if calculate_xy_from_bodyparts:
        out = calculate_body_center(
            out,
            belly_x_col=belly_x_col,
            belly_y_col=belly_y_col,
            sternum_x_col=sternum_x_col,
            sternum_y_col=sternum_y_col,
            x_col=x_col,
            y_col=y_col,
        )
    else:
        _require_columns(out, [x_col, y_col])
        out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
        out[y_col] = pd.to_numeric(out[y_col], errors="coerce")

    if circle_info is None:
        if not estimate_circle_if_missing:
            raise ValueError(
                "circle_info is required when estimate_circle_if_missing=False."
            )

        if mouse_col in out.columns:
            circle_info = {
                mouse_id: estimate_circle_from_extremes(group, x_col=x_col, y_col=y_col)
                for mouse_id, group in out.groupby(mouse_col, sort=False, dropna=False)
            }
        else:
            circle_info = estimate_circle_from_extremes(out, x_col=x_col, y_col=y_col)

    out = calculate_distance_from_wall(
        out,
        circle_info=circle_info,
        x_col=x_col,
        y_col=y_col,
        mouse_col=mouse_col,
    )
    out = calculate_movement_angle_to_wall(
        out,
        belly_x_col=belly_x_col,
        belly_y_col=belly_y_col,
        sternum_x_col=sternum_x_col,
        sternum_y_col=sternum_y_col,
        x_col=x_col,
        y_col=y_col,
    )
    out = calculate_kde(
        out,
        x_col=x_col,
        y_col=y_col,
        mouse_col=mouse_col,
        normalize_0_to_1=normalize_kde_0_to_1,
    )

    hmm_state = pd.to_numeric(out[hmm_state_col], errors="coerce")
    finite_state = hmm_state.dropna()
    noninteger = ~np.isclose(finite_state, np.round(finite_state))
    if np.any(noninteger):
        raise ValueError(f"{hmm_state_col!r} contains non-integer state values.")

    observed = set(finite_state.round().astype(int).unique())
    unexpected = observed.difference({1, 2})
    if unexpected:
        raise ValueError(
            f"Unexpected Level-1 states {sorted(unexpected)}; expected 1 and 2."
        )

    out["HMM_State"] = hmm_state.round().astype("Int64")
    return out


def create_level2_datastreams_dict(
    dict_all_df: Mapping[Any, pd.DataFrame],
    circle_info_by_mouse: Mapping[Any, Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[Any, pd.DataFrame]:
    """Run ``create_level2_datastreams`` separately for every mouse."""
    output: dict[Any, pd.DataFrame] = {}

    for mouse_id, df_mouse in dict_all_df.items():
        circle = None
        if circle_info_by_mouse is not None:
            if mouse_id in circle_info_by_mouse:
                circle = circle_info_by_mouse[mouse_id]
            elif str(mouse_id) in circle_info_by_mouse:
                circle = circle_info_by_mouse[str(mouse_id)]
            else:
                raise ValueError(f"No circle information found for mouse {mouse_id!r}.")

        output[mouse_id] = create_level2_datastreams(
            df=df_mouse,
            circle_info=circle,
            **kwargs,
        )

    return output

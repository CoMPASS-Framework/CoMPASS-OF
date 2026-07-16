"""Datastream preparation utilities for the Level-1 HMM.

This module computes session-wise step lengths and turning angles from ordered
position data and provides a diagnostic distribution plot.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = [
    "prep_data",
    "plot_step_and_angle_distributions",
]


# =========================================================
# Geometry helpers
# =========================================================
def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters for scalar or vector NumPy arrays."""
    radius_m = 6_371_000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_m * c


def _euclid(x1, y1, x2, y2):
    """Euclidean distance for scalar or vector NumPy arrays."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _turn_angle(p0, p1, p2):
    """Return the signed turning angle at ``p1`` in radians, in (-pi, pi]."""
    v1 = p1 - p0
    v2 = p2 - p1

    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return np.nan

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan

    v1u = v1 / n1
    v2u = v2 / n2

    dot = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
    angle = np.arccos(dot)

    # Sign from the z-component of the 2D cross product.
    cross_z = v1u[0] * v2u[1] - v1u[1] * v2u[0]
    return np.sign(cross_z) * angle


def _dist_angle(prev_xy, cur_xy, target_xy, coord_type):
    """Compute current-to-target distance and prev-current-target turn angle."""
    if coord_type == "LL":
        # x = longitude, y = latitude
        distance = _haversine_m(
            cur_xy[1],
            cur_xy[0],
            target_xy[1],
            target_xy[0],
        )
    else:
        distance = _euclid(
            cur_xy[0],
            cur_xy[1],
            target_xy[0],
            target_xy[1],
        )

    angle = _turn_angle(
        np.asarray(prev_xy, dtype=float),
        np.asarray(cur_xy, dtype=float),
        np.asarray(target_xy, dtype=float),
    )
    return distance, angle


# =========================================================
# Main datastream preparation
# =========================================================
def prep_data(
    data: pd.DataFrame,
    type: Literal["UTM", "LL"] = "UTM",
    coordNames: tuple[str, str] = ("x", "y"),
    covNames: list[str] | None = None,
    centers: np.ndarray | None = None,
    centroids: dict | None = None,
    angleCovs: list[str] | None = None,
    altCoordNames: str | None = None,
) -> pd.DataFrame:
    """Prepare ordered movement datastreams for the Level-1 HMM.

    The function sorts observations by ``Session`` and ``S_no``, creates
    ``ID = Session``, and computes session-wise ``step`` and ``angle`` values.
    It does not fill covariates or remove rows containing missing values.

    Parameters
    ----------
    data
        Input table containing ``Session``, ``S_no``, and coordinate columns.
    type
        ``"UTM"`` for planar Euclidean coordinates or ``"LL"`` for
        longitude/latitude coordinates.
    coordNames
        Names of the x and y coordinate columns.
    covNames
        Covariate columns retained in the arranged output.
    centers
        Optional fixed center coordinates with shape ``(K, 2)``.
    centroids
        Optional mapping from centroid name to a DataFrame containing ``x``,
        ``y``, and exactly one time-key column.
    angleCovs
        Angle covariate columns retained in the arranged output.
    altCoordNames
        Optional base name for copied coordinate columns, producing
        ``<base>.x`` and ``<base>.y``.

    Returns
    -------
    pandas.DataFrame
        Input data with computed datastreams and optional center/centroid
        distance and angle features.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data is empty")
    if len(coordNames) != 2:
        raise ValueError("coordNames must contain exactly two column names")

    xcol, ycol = coordNames
    missing_required = [
        column
        for column in ("Session", "S_no", xcol, ycol)
        if column not in data.columns
    ]
    if missing_required:
        raise ValueError(
            "Required columns not found in data: " + ", ".join(missing_required)
        )

    coord_type = type.upper()
    if coord_type not in {"UTM", "LL"}:
        raise ValueError("type must be either 'UTM' or 'LL'")

    df = data.copy()
    df["ID"] = df["Session"].astype(str)

    # Keep each session contiguous and preserve temporal ordering.
    df["S_no"] = pd.to_numeric(df["S_no"], errors="coerce")
    df = df.sort_values(["Session", "S_no"], kind="mergesort").reset_index(
        drop=True
    )

    out_x, out_y = ("x", "y")
    if altCoordNames:
        out_x, out_y = f"{altCoordNames}.x", f"{altCoordNames}.y"

    df[out_x] = df[xcol]
    df[out_y] = df[ycol]

    df["step"] = np.nan
    df["angle"] = np.nan

    covNames = [] if covNames is None else list(dict.fromkeys(covNames))
    angleCovs = [] if angleCovs is None else list(dict.fromkeys(angleCovs))
    cov_all = list(dict.fromkeys(covNames + angleCovs))

    for column in cov_all:
        if column not in df.columns:
            raise ValueError(f"covariate '{column}' not found in data")

    # Compute step and angle within each session in sorted row order.
    for _, group in df.groupby("ID", sort=False, observed=False):
        index = group.index
        x = group[xcol].to_numpy(dtype=float)
        y = group[ycol].to_numpy(dtype=float)

        step = np.full(len(group), np.nan)
        if len(group) > 1:
            if coord_type == "LL":
                step[1:] = _haversine_m(y[:-1], x[:-1], y[1:], x[1:])
            else:
                step[1:] = _euclid(x[:-1], y[:-1], x[1:], y[1:])
        df.loc[index, "step"] = step

        # R-style shifted alignment: angle at (k-1, k, k+1) is stored at k+1.
        angle = np.full(len(group), np.nan)
        for k in range(1, len(group) - 1):
            angle[k + 1] = _turn_angle(
                np.array([x[k - 1], y[k - 1]]),
                np.array([x[k], y[k]]),
                np.array([x[k + 1], y[k + 1]]),
            )
        df.loc[index, "angle"] = angle

    # Fixed centers.
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers must be a (K, 2) matrix")

        for j in range(centers.shape[0]):
            df[f"center{j + 1}.dist"] = np.nan
            df[f"center{j + 1}.angle"] = np.nan

        for _, group in df.groupby("ID", sort=False, observed=False):
            index = group.index
            x = group[xcol].to_numpy(dtype=float)
            y = group[ycol].to_numpy(dtype=float)

            for j in range(centers.shape[0]):
                distances = np.full(len(group), np.nan)
                angles = np.full(len(group), np.nan)

                for k in range(1, len(group)):
                    distance, angle = _dist_angle(
                        (x[k - 1], y[k - 1]),
                        (x[k], y[k]),
                        (centers[j, 0], centers[j, 1]),
                        coord_type,
                    )
                    distances[k] = distance
                    angles[k] = angle

                df.loc[index, f"center{j + 1}.dist"] = distances
                df.loc[index, f"center{j + 1}.angle"] = angles

    # Time-varying centroids.
    if centroids is not None:
        if not isinstance(centroids, dict):
            raise ValueError(
                "centroids must be a dict of name -> DataFrame([x, y, time_col])"
            )

        for name, centroid_df in centroids.items():
            if not isinstance(centroid_df, pd.DataFrame):
                raise ValueError(f"centroid '{name}' must be a pandas DataFrame")
            if not {"x", "y"}.issubset(centroid_df.columns):
                raise ValueError(
                    f"centroid '{name}' must contain x and y columns"
                )

            time_columns = [
                column for column in centroid_df.columns if column not in ("x", "y")
            ]
            if len(time_columns) != 1:
                raise ValueError(
                    f"centroid '{name}' must have exactly one time column"
                )

            time_column = time_columns[0]
            if time_column not in df.columns:
                raise ValueError(
                    f"time column '{time_column}' for centroid '{name}' "
                    "not found in data"
                )

            centroid_x = f"__{name}_x"
            centroid_y = f"__{name}_y"
            centroid_use = centroid_df.rename(
                columns={"x": centroid_x, "y": centroid_y}
            )
            df = df.merge(centroid_use, how="left", on=time_column)

            distance_column = f"{name}.dist"
            angle_column = f"{name}.angle"
            df[distance_column] = np.nan
            df[angle_column] = np.nan

            for _, group in df.groupby("ID", sort=False, observed=False):
                index = group.index
                x = group[xcol].to_numpy(dtype=float)
                y = group[ycol].to_numpy(dtype=float)
                cx = group[centroid_x].to_numpy(dtype=float)
                cy = group[centroid_y].to_numpy(dtype=float)

                distances = np.full(len(group), np.nan)
                angles = np.full(len(group), np.nan)

                for k in range(1, len(group)):
                    if np.isnan(cx[k]) or np.isnan(cy[k]):
                        continue

                    distance, angle = _dist_angle(
                        (x[k - 1], y[k - 1]),
                        (x[k], y[k]),
                        (cx[k], cy[k]),
                        coord_type,
                    )
                    distances[k] = distance
                    angles[k] = angle

                df.loc[index, distance_column] = distances
                df.loc[index, angle_column] = angles

            df.drop(columns=[centroid_x, centroid_y], inplace=True)

    base_columns = ["ID", "step", "angle"]
    extra_columns = [
        column
        for column in df.columns
        if column.endswith(".dist") or column.endswith(".angle")
    ]
    coordinate_columns = [out_x, out_y]

    ordered = list(
        dict.fromkeys(base_columns + cov_all + extra_columns + coordinate_columns)
    )
    remainder = [column for column in df.columns if column not in ordered]
    df = df[ordered + remainder]

    df["ID"] = df["ID"].astype("category")
    df.attrs["coords"] = coordinate_columns
    return df


# =========================================================
# Datastream diagnostics
# =========================================================
def plot_step_and_angle_distributions(df: pd.DataFrame) -> plt.Figure:
    """Plot step-length and turning-angle histograms and return the figure."""
    required = {"step", "angle"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    step = pd.to_numeric(df["step"], errors="coerce").dropna()
    angle = pd.to_numeric(df["angle"], errors="coerce").dropna()

    if step.empty:
        raise ValueError("No finite step values are available for plotting")
    if angle.empty:
        raise ValueError("No finite angle values are available for plotting")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    step_max = float(step.max())
    step_stop = max(20, int(np.ceil(step_max / 20.0) * 20) + 20)
    sns.histplot(
        step,
        bins=range(0, step_stop, 20),
        kde=False,
        ax=ax1,
    )
    ax1.set_title("Step Length Distribution")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Count")

    angle_range = float(angle.max() - angle.min())
    angle_bins = max(1, int(np.ceil(angle_range / 0.1)))
    sns.histplot(
        angle,
        bins=angle_bins,
        kde=False,
        ax=ax2,
    )
    ax2.set_title("Turning Angle Distribution")
    ax2.set_xlabel("Angle (rad)")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    return fig

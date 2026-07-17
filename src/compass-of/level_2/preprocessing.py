"""Data preparation and sequence utilities for the CoMPASS Level-2 model.

This module contains row ordering, feature validation, continuous-feature
standardization, categorical one-hot encoding, invalid-row filtering, session
splitting, and forward-chaining split generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FeaturePreprocessor",
    "prepare_mouse_df",
    "validate_features",
    "fit_feature_preprocessor",
    "transform_features",
    "drop_invalid_feature_rows",
    "split_sessions_dict",
    "concat_sequences",
    "fit_preproc_and_transform_train_blocks",
    "generate_forward_chaining_splits",
]

@dataclass
class FeaturePreprocessor:
    continuous_features: list[str]
    categorical_features: list[str]
    use_standardization: bool
    scaler: Any
    dummy_columns: list[str]
    final_feature_names: list[str]


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare_mouse_df(
    df_mouse: pd.DataFrame,
    session_col_in: str = "Session",
    mouse_col_in: str = "MouseID",
    order_candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    if order_candidates is None:
        order_candidates = ["S_no", "frame"]

    df = df_mouse.copy()
    df["_orig_row_id"] = np.arange(len(df))

    rename_map = {}
    if session_col_in in df.columns and "session" not in df.columns:
        rename_map[session_col_in] = "session"
    if mouse_col_in in df.columns and "mouseid" not in df.columns:
        rename_map[mouse_col_in] = "mouseid"

    df = df.rename(columns=rename_map)

    if "session" not in df.columns:
        raise ValueError("No session column found.")

    order_col = None
    for c in order_candidates:
        if c in df.columns:
            order_col = c
            break

    if order_col is None:
        raise ValueError(f"No order column found. Tried {order_candidates}")

    df = df.sort_values(["session", order_col]).reset_index(drop=True)
    return df, order_col


def validate_features(df: pd.DataFrame, features: list[str]) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    bad = [f for f in features if df[f].isna().all()]
    if bad:
        raise ValueError(f"These features are entirely NaN: {bad}")


def _default_categorical_features(features: list[str]) -> list[str]:
    out = []
    for f in features:
        if f.lower() in {"hmm_state", "state", "level_1_states", "level1_state"}:
            out.append(f)
    return out


def fit_feature_preprocessor(
    df_train: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None = None,
    use_standardization: bool = True,
) -> FeaturePreprocessor:
    if categorical_features is None:
        categorical_features = _default_categorical_features(features)

    categorical_features = [f for f in categorical_features if f in features]
    continuous_features = [f for f in features if f not in categorical_features]

    train = df_train.copy()

    cont_df = pd.DataFrame(index=train.index)
    for f in continuous_features:
        cont_df[f] = _safe_numeric_series(train[f])

    if cont_df.isna().any().any():
        raise ValueError("NaNs found in continuous training features after numeric coercion.")

    scaler = None
    if use_standardization and len(continuous_features) > 0:
        scaler = StandardScaler()
        cont_values = scaler.fit_transform(cont_df.values)
    else:
        cont_values = cont_df.values.astype(float)

    if len(categorical_features) > 0:
        cat_df = pd.DataFrame(index=train.index)
        for f in categorical_features:
            cat_df[f] = train[f].astype("Int64").astype(str)

        dummies = pd.get_dummies(cat_df, prefix=categorical_features, dtype=float)
        dummy_columns = list(dummies.columns)
        cat_values = dummies.values.astype(float)
    else:
        dummy_columns = []
        cat_values = np.empty((len(train), 0), dtype=float)

    final_feature_names = list(continuous_features) + dummy_columns

    return FeaturePreprocessor(
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
        scaler=scaler,
        dummy_columns=dummy_columns,
        final_feature_names=final_feature_names,
    )


def transform_features(df: pd.DataFrame, preproc: FeaturePreprocessor) -> np.ndarray:
    work = df.copy()

    if len(preproc.continuous_features) > 0:
        cont_df = pd.DataFrame(index=work.index)
        for f in preproc.continuous_features:
            cont_df[f] = _safe_numeric_series(work[f])

        if cont_df.isna().any().any():
            raise ValueError("NaNs found in continuous features during transform.")

        if preproc.use_standardization:
            cont_values = preproc.scaler.transform(cont_df.values)
        else:
            cont_values = cont_df.values.astype(float)
    else:
        cont_values = np.empty((len(work), 0), dtype=float)

    if len(preproc.categorical_features) > 0:
        cat_df = pd.DataFrame(index=work.index)
        for f in preproc.categorical_features:
            cat_df[f] = work[f].astype("Int64").astype(str)

        dummies = pd.get_dummies(cat_df, prefix=preproc.categorical_features, dtype=float)
        dummies = dummies.reindex(columns=preproc.dummy_columns, fill_value=0.0)
        cat_values = dummies.values.astype(float)
    else:
        cat_values = np.empty((len(work), 0), dtype=float)

    return np.hstack([cont_values, cat_values]).astype(float)


def drop_invalid_feature_rows(
    df: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    if categorical_features is None:
        categorical_features = _default_categorical_features(features)

    categorical_features = [f for f in categorical_features if f in features]
    continuous_features = [f for f in features if f not in categorical_features]

    out = df.copy()

    for f in continuous_features:
        out[f] = _safe_numeric_series(out[f])

    keep_mask = pd.Series(True, index=out.index)

    if len(continuous_features) > 0:
        keep_mask &= out[continuous_features].notna().all(axis=1)

    if len(categorical_features) > 0:
        keep_mask &= out[categorical_features].notna().all(axis=1)

    return out.loc[keep_mask].copy()


# ============================================================
# Sequence helpers
# ============================================================

def split_sessions_dict(df_mouse: pd.DataFrame) -> dict[Any, pd.DataFrame]:
    return {sess: g.copy() for sess, g in df_mouse.groupby("session", sort=False)}


def concat_sequences(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [d for d in dfs if isinstance(d, pd.DataFrame) and len(d) > 0]
    if len(valid) == 0:
        return pd.DataFrame()
    return pd.concat(valid, axis=0, ignore_index=False)


def fit_preproc_and_transform_train_blocks(
    train_blocks: list[pd.DataFrame],
    features: list[str],
    categorical_features: list[str] | None,
    use_standardization: bool,
) -> tuple[FeaturePreprocessor, np.ndarray, list[int]]:
    df_train_concat = concat_sequences(train_blocks)
    if df_train_concat.empty:
        raise ValueError("No training blocks available.")

    preproc = fit_feature_preprocessor(
        df_train=df_train_concat,
        features=features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
    )

    X_blocks = []
    lengths = []

    for blk in train_blocks:
        if len(blk) == 0:
            continue
        X_blk = transform_features(blk, preproc)
        X_blocks.append(X_blk)
        lengths.append(len(X_blk))

    if len(X_blocks) == 0:
        raise ValueError("No valid transformed training blocks.")

    X_train = np.vstack(X_blocks)
    return preproc, X_train, lengths


# ============================================================
# Time-aware helpers
# ============================================================

def generate_forward_chaining_splits(
    n_rows: int,
    n_splits: int = 3,
    min_train_size: int = 150,
    min_val_size: int = 60,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < (min_train_size + gap + min_val_size):
        return []

    if n_splits < 1:
        n_splits = 1

    usable_tail = n_rows - min_train_size - gap
    if usable_tail < min_val_size:
        return []

    test_size = max(min_val_size, usable_tail // (n_splits + 1))
    splits = []

    train_end = min_train_size
    while True:
        val_start = train_end + gap
        val_end = val_start + test_size

        if val_end > n_rows:
            break

        tr_idx = np.arange(0, train_end)
        va_idx = np.arange(val_start, val_end)

        if len(tr_idx) >= min_train_size and len(va_idx) >= min_val_size:
            splits.append((tr_idx, va_idx))

        train_end += test_size
        if len(splits) >= n_splits:
            break

    return splits

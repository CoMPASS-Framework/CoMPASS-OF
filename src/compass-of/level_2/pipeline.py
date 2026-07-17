"""Outer evaluation and final per-mouse pipeline for CoMPASS Level-2.

The main public entry points are:

- ``run_level2_bgmm_gmmhmm_timeaware_one_mouse``
- ``run_level2_bgmm_gmmhmm_timeaware_dict``

Importing this module does not automatically fit a model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

try:
    from .level2_preprocessing import (
        FeaturePreprocessor,
        concat_sequences,
        drop_invalid_feature_rows,
        fit_preproc_and_transform_train_blocks,
        prepare_mouse_df,
        split_sessions_dict,
        transform_features,
        validate_features,
    )
    from .level2_model import (
        _compute_aic_from_loglik,
        choose_best_hyperparams_nested,
        fit_one_candidate_with_restarts,
    )
except ImportError:
    from level2_preprocessing import (
        FeaturePreprocessor,
        concat_sequences,
        drop_invalid_feature_rows,
        fit_preproc_and_transform_train_blocks,
        prepare_mouse_df,
        split_sessions_dict,
        transform_features,
        validate_features,
    )
    from level2_model import (
        _compute_aic_from_loglik,
        choose_best_hyperparams_nested,
        fit_one_candidate_with_restarts,
    )

__all__ = [
    "fit_model_on_session_blocks",
    "evaluate_one_outer_fold",
    "fit_final_model_all_sessions",
    "run_level2_bgmm_gmmhmm_timeaware_one_mouse",
    "run_level2_bgmm_gmmhmm_timeaware_dict",
]

def fit_model_on_session_blocks(
    train_session_blocks: list[pd.DataFrame],
    features: list[str],
    categorical_features: list[str] | None,
    use_standardization: bool,
    n_states: int,
    n_mix: int,
    reg_val: float,
    restart_seeds: list[int],
    n_iter: int = 400,
    verbose: bool = False,
) -> tuple[GMMHMM, FeaturePreprocessor, np.ndarray, list[int]]:
    preproc, X_train, lengths_train = fit_preproc_and_transform_train_blocks(
        train_blocks=train_session_blocks,
        features=features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
    )

    model, _ = fit_one_candidate_with_restarts(
        X_train=X_train,
        lengths_train=lengths_train,
        n_states=n_states,
        n_mix=n_mix,
        reg_val=reg_val,
        restart_seeds=restart_seeds,
        n_iter=n_iter,
        verbose=verbose,
    )

    return model, preproc, X_train, lengths_train


def _compute_state_occupancy_from_preds(preds: np.ndarray, n_states: int) -> np.ndarray:
    occ = np.bincount(preds, minlength=n_states).astype(float)
    denom = occ.sum()
    if denom <= 0:
        return np.zeros(n_states, dtype=float)
    return occ / denom


def evaluate_one_outer_fold(
    train_session_dict: dict[Any, pd.DataFrame],
    test_session_key: Any,
    test_session_df: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None,
    use_standardization: bool,
    n_states: int,
    n_mix_options: list[int],
    reg_options: list[float],
    n_splits_options: list[int],
    gap: int,
    min_train_size: int,
    min_val_size: int,
    restart_seeds: list[int],
    n_iter: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    best_params, tuning_df = choose_best_hyperparams_nested(
        session_dict_train=train_session_dict,
        features=features,
        categorical_features=categorical_features,
        n_states=n_states,
        n_mix_options=n_mix_options,
        reg_options=reg_options,
        n_splits_options=n_splits_options,
        gap=gap,
        min_train_size=min_train_size,
        min_val_size=min_val_size,
        use_standardization=use_standardization,
        restart_seeds=restart_seeds,
        n_iter=n_iter,
        verbose=verbose,
    )

    train_blocks = [train_session_dict[s].copy() for s in train_session_dict.keys()]

    model, preproc, X_train, lengths_train = fit_model_on_session_blocks(
        train_session_blocks=train_blocks,
        features=features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
        n_states=best_params["n_states"],
        n_mix=best_params["n_mix"],
        reg_val=best_params["reg_val"],
        restart_seeds=restart_seeds,
        n_iter=n_iter,
        verbose=verbose,
    )

    X_test = transform_features(test_session_df, preproc)
    test_ll_total = model.score(X_test, lengths=[len(X_test)])
    test_ll_per_sample = test_ll_total / max(len(X_test), 1)

    posteriors = model.predict_proba(X_test, lengths=[len(X_test)])
    preds = posteriors.argmax(axis=1)
    avg_entropy = float(
        -np.mean(np.sum(posteriors * np.log(np.clip(posteriors, 1e-12, 1.0)), axis=1))
    )
    mean_certainty = float(np.mean(np.max(posteriors, axis=1)))

    test_aic = _compute_aic_from_loglik(
        loglik_total=test_ll_total,
        model=model,
        n_features=X_train.shape[1],
    )

    occ = _compute_state_occupancy_from_preds(preds=preds, n_states=n_states)

    pred_df = test_session_df.copy()
    pred_df["Level_2_States_OOF"] = preds
    for s in range(n_states):
        pred_df[f"Level_2_PostProb_OOF_{s}"] = posteriors[:, s]

    meta = {
        "test_session": test_session_key,
        "error": None,
        "best_n_states": best_params["n_states"],
        "best_n_mix": best_params["n_mix"],
        "best_reg_val": best_params["reg_val"],
        "inner_strategy": best_params["inner_strategy"],
        "test_loglik_total": float(test_ll_total),
        "test_loglik_per_sample": float(test_ll_per_sample),
        "test_aic": float(test_aic),
        "test_avg_posterior_entropy": avg_entropy,
        "test_mean_certainty": mean_certainty,
        "n_train_samples": int(sum(lengths_train)),
        "n_test_samples": int(len(X_test)),
        "tuning_table": tuning_df,
    }

    for s in range(n_states):
        meta[f"state_occupancy_{s}"] = float(occ[s])

    return pred_df, meta


def fit_final_model_all_sessions(
    session_dict_all: dict[Any, pd.DataFrame],
    features: list[str],
    categorical_features: list[str] | None,
    use_standardization: bool,
    n_states: int,
    n_mix_options: list[int],
    reg_options: list[float],
    n_splits_options: list[int],
    gap: int,
    min_train_size: int,
    min_val_size: int,
    restart_seeds: list[int],
    n_iter: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    best_params, tuning_df = choose_best_hyperparams_nested(
        session_dict_train=session_dict_all,
        features=features,
        categorical_features=categorical_features,
        n_states=n_states,
        n_mix_options=n_mix_options,
        reg_options=reg_options,
        n_splits_options=n_splits_options,
        gap=gap,
        min_train_size=min_train_size,
        min_val_size=min_val_size,
        use_standardization=use_standardization,
        restart_seeds=restart_seeds,
        n_iter=n_iter,
        verbose=verbose,
    )

    all_blocks = [session_dict_all[s].copy() for s in session_dict_all.keys()]

    model, preproc, X_all, lengths_all = fit_model_on_session_blocks(
        train_session_blocks=all_blocks,
        features=features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
        n_states=best_params["n_states"],
        n_mix=best_params["n_mix"],
        reg_val=best_params["reg_val"],
        restart_seeds=restart_seeds,
        n_iter=n_iter,
        verbose=verbose,
    )

    final_loglik_total = float(model.score(X_all, lengths=lengths_all))
    final_loglik_per_sample = float(final_loglik_total / max(len(X_all), 1))
    final_aic = _compute_aic_from_loglik(
        loglik_total=final_loglik_total,
        model=model,
        n_features=X_all.shape[1],
    )

    posteriors = model.predict_proba(X_all, lengths=lengths_all)
    preds = posteriors.argmax(axis=1)

    df_all = concat_sequences(all_blocks).copy()
    df_all["Level_2_States"] = preds
    for s in range(n_states):
        df_all[f"Level_2_PostProb_{s}"] = posteriors[:, s]

    occ = _compute_state_occupancy_from_preds(preds=preds, n_states=n_states)
    final_avg_posterior_entropy = float(
        -np.mean(np.sum(posteriors * np.log(np.clip(posteriors, 1e-12, 1.0)), axis=1))
    )
    final_mean_certainty = float(np.mean(np.max(posteriors, axis=1)))

    meta = {
        "best_n_states": best_params["n_states"],
        "best_n_mix": best_params["n_mix"],
        "best_reg_val": best_params["reg_val"],
        "inner_strategy": best_params["inner_strategy"],
        "final_loglik_total": final_loglik_total,
        "final_loglik_per_sample": final_loglik_per_sample,
        "final_aic": final_aic,
        "final_avg_posterior_entropy": final_avg_posterior_entropy,
        "final_mean_certainty": final_mean_certainty,
        "tuning_table": tuning_df,
        "transmat_": model.transmat_.copy(),
        "startprob_": model.startprob_.copy(),
        "feature_names_used": preproc.final_feature_names,
        "lengths_all": lengths_all,
        "model": model,
        "preprocessor": preproc,
    }

    for s in range(n_states):
        meta[f"state_occupancy_{s}"] = float(occ[s])

    return df_all, meta


# ============================================================
# Per-mouse pipeline
# ============================================================

def run_level2_bgmm_gmmhmm_timeaware_one_mouse(
    df_mouse: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None = None,
    n_states: int = 3,
    n_mix_options: list[int] | None = None,
    reg_options: list[float] | None = None,
    n_splits_options: list[int] | None = None,
    gap: int = 0,
    min_train_size: int = 150,
    min_val_size: int = 60,
    use_standardization: bool = True,
    restart_seeds: list[int] | None = None,
    n_iter: int = 400,
    session_col_in: str = "Session",
    mouse_col_in: str = "MouseID",
    order_candidates: list[str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if n_mix_options is None:
        n_mix_options = [1, 2]
    if reg_options is None:
        reg_options = [1e-3, 1e-4]
    if n_splits_options is None:
        n_splits_options = [3, 4]
    if restart_seeds is None:
        restart_seeds = [3, 7, 11, 19, 23, 29, 31, 37]

    df_prepped, order_col = prepare_mouse_df(
        df_mouse=df_mouse,
        session_col_in=session_col_in,
        mouse_col_in=mouse_col_in,
        order_candidates=order_candidates,
    )

    validate_features(df_prepped, features)

    if categorical_features is None:
        categorical_features = _default_categorical_features(features)

    df_valid = drop_invalid_feature_rows(
        df_prepped,
        features=features,
        categorical_features=categorical_features,
    ).copy()

    if df_valid.empty:
        raise ValueError("No valid rows remain after filtering features.")

    session_dict = split_sessions_dict(df_valid)
    session_keys = list(session_dict.keys())

    outer_pred_parts = []
    outer_meta = []

    if len(session_keys) >= 2:
        for heldout_sess in session_keys:
            train_session_dict = {
                s: session_dict[s].copy()
                for s in session_keys if s != heldout_sess
            }
            test_session_df = session_dict[heldout_sess].copy()

            try:
                pred_df, meta = evaluate_one_outer_fold(
                    train_session_dict=train_session_dict,
                    test_session_key=heldout_sess,
                    test_session_df=test_session_df,
                    features=features,
                    categorical_features=categorical_features,
                    use_standardization=use_standardization,
                    n_states=n_states,
                    n_mix_options=n_mix_options,
                    reg_options=reg_options,
                    n_splits_options=n_splits_options,
                    gap=gap,
                    min_train_size=min_train_size,
                    min_val_size=min_val_size,
                    restart_seeds=restart_seeds,
                    n_iter=n_iter,
                    verbose=verbose,
                )
                outer_pred_parts.append(pred_df)
                outer_meta.append({k: v for k, v in meta.items() if k != "tuning_table"})

                if verbose:
                    print(
                        f"[outer] heldout session={heldout_sess} | "
                        f"ll/sample={meta['test_loglik_per_sample']:.4f} | "
                        f"AIC={meta['test_aic']:.2f} | "
                        f"mix={meta['best_n_mix']} reg={meta['best_reg_val']:.0e} | "
                        f"inner={meta['inner_strategy']}"
                    )

            except Exception as e:
                row = {
                    "error": str(e),
                    "test_session": heldout_sess,
                    "best_n_states": None,
                    "best_n_mix": None,
                    "best_reg_val": None,
                    "inner_strategy": None,
                    "test_loglik_total": np.nan,
                    "test_loglik_per_sample": np.nan,
                    "test_aic": np.nan,
                    "test_avg_posterior_entropy": np.nan,
                    "test_mean_certainty": np.nan,
                    "n_train_samples": np.nan,
                    "n_test_samples": len(test_session_df),
                }
                for s in range(n_states):
                    row[f"state_occupancy_{s}"] = np.nan
                outer_meta.append(row)

                if verbose:
                    print(f"[outer failed] heldout session={heldout_sess} | {e}")

        if len(outer_pred_parts) > 0:
            df_oof = pd.concat(outer_pred_parts, axis=0).sort_values(["session", order_col]).reset_index(drop=True)
        else:
            df_oof = pd.DataFrame()
    else:
        df_oof = pd.DataFrame()
        row = {
            "error": "Only one session available; outer LOSO evaluation not possible.",
            "test_session": None,
            "best_n_states": None,
            "best_n_mix": None,
            "best_reg_val": None,
            "inner_strategy": None,
            "test_loglik_total": np.nan,
            "test_loglik_per_sample": np.nan,
            "test_aic": np.nan,
            "test_avg_posterior_entropy": np.nan,
            "test_mean_certainty": np.nan,
            "n_train_samples": np.nan,
            "n_test_samples": np.nan,
        }
        for s in range(n_states):
            row[f"state_occupancy_{s}"] = np.nan
        outer_meta.append(row)

    df_final_valid, final_meta = fit_final_model_all_sessions(
        session_dict_all=session_dict,
        features=features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
        n_states=n_states,
        n_mix_options=n_mix_options,
        reg_options=reg_options,
        n_splits_options=n_splits_options,
        gap=gap,
        min_train_size=min_train_size,
        min_val_size=min_val_size,
        restart_seeds=restart_seeds,
        n_iter=n_iter,
        verbose=verbose,
    )

    df_out = df_prepped.copy()
    final_cols = ["Level_2_States"] + [f"Level_2_PostProb_{s}" for s in range(n_states)]
    for c in final_cols:
        df_out[c] = np.nan

    aligned_final = df_final_valid[["_orig_row_id"] + final_cols].copy()
    df_out = df_out.merge(aligned_final, on="_orig_row_id", how="left", suffixes=("", "_new"))

    for c in final_cols:
        if f"{c}_new" in df_out.columns:
            df_out[c] = df_out[f"{c}_new"]
            df_out.drop(columns=[f"{c}_new"], inplace=True)

    if not df_oof.empty:
        oof_cols = ["Level_2_States_OOF"] + [f"Level_2_PostProb_OOF_{s}" for s in range(n_states)]
        for c in oof_cols:
            if c not in df_out.columns:
                df_out[c] = np.nan

        aligned_oof = df_oof[["_orig_row_id"] + oof_cols].copy()
        df_out = df_out.merge(aligned_oof, on="_orig_row_id", how="left", suffixes=("", "_oofnew"))

        for c in oof_cols:
            if f"{c}_oofnew" in df_out.columns:
                df_out[c] = df_out[f"{c}_oofnew"]
                df_out.drop(columns=[f"{c}_oofnew"], inplace=True)

    if "session" in df_out.columns and "Session" not in df_out.columns:
        df_out["Session"] = df_out["session"]
    if "mouseid" in df_out.columns and "MouseID" not in df_out.columns:
        df_out["MouseID"] = df_out["mouseid"]

    outer_cv_table = pd.DataFrame(outer_meta)

    meta = {
        "order_col_used": order_col,
        "n_rows_original": len(df_prepped),
        "n_rows_valid_for_modeling": len(df_valid),
        "n_rows_with_final_predictions": int(df_out["Level_2_States"].notna().sum()),
        "n_rows_with_oof_predictions": int(df_out["Level_2_States_OOF"].notna().sum()) if "Level_2_States_OOF" in df_out.columns else 0,
        "n_sessions": int(df_valid["session"].nunique()),
        "outer_cv_table": outer_cv_table,
        "outer_cv_mean_loglik_per_sample": float(
            outer_cv_table["test_loglik_per_sample"].dropna().mean()
        ) if not outer_cv_table.empty else np.nan,
        "outer_cv_sd_loglik_per_sample": float(
            outer_cv_table["test_loglik_per_sample"].dropna().std(ddof=0)
        ) if not outer_cv_table.empty else np.nan,
        "outer_cv_mean_aic": float(
            outer_cv_table["test_aic"].dropna().mean()
        ) if ("test_aic" in outer_cv_table.columns and not outer_cv_table.empty) else np.nan,
        "final_model_meta": final_meta,
        "features_used": features,
        "categorical_features_used": categorical_features,
        "use_standardization": use_standardization,
    }

    return df_out, meta


def run_level2_bgmm_gmmhmm_timeaware_dict(
    dict_all_df: dict,
    features: list[str],
    categorical_features: list[str] | None = None,
    n_states: int = 3,
    n_mix_options: list[int] | None = None,
    reg_options: list[float] | None = None,
    n_splits_options: list[int] | None = None,
    gap: int = 0,
    min_train_size: int = 150,
    min_val_size: int = 60,
    use_standardization: bool = True,
    restart_seeds: list[int] | None = None,
    n_iter: int = 400,
    session_col_in: str = "Session",
    mouse_col_in: str = "MouseID",
    order_candidates: list[str] | None = None,
    verbose: bool = False,
) -> tuple[dict, dict]:
    dict_level2 = {}
    meta_by_mouse = {}

    for mouse_key, df_mouse in dict_all_df.items():
        print(f"\n================ Mouse {mouse_key} ================\n")

        try:
            df_out, meta = run_level2_bgmm_gmmhmm_timeaware_one_mouse(
                df_mouse=df_mouse,
                features=features,
                categorical_features=categorical_features,
                n_states=n_states,
                n_mix_options=n_mix_options,
                reg_options=reg_options,
                n_splits_options=n_splits_options,
                gap=gap,
                min_train_size=min_train_size,
                min_val_size=min_val_size,
                use_standardization=use_standardization,
                restart_seeds=restart_seeds,
                n_iter=n_iter,
                session_col_in=session_col_in,
                mouse_col_in=mouse_col_in,
                order_candidates=order_candidates,
                verbose=verbose,
            )

            dict_level2[mouse_key] = df_out
            meta_by_mouse[mouse_key] = {"error": None, **meta}

            if pd.notna(meta["outer_cv_mean_loglik_per_sample"]):
                print(
                    f"Done mouse {mouse_key} | "
                    f"predicted_rows={meta['n_rows_with_final_predictions']}/{meta['n_rows_original']} | "
                    f"oof_rows={meta['n_rows_with_oof_predictions']} | "
                    f"sessions={meta['n_sessions']} | "
                    f"outer_ll/sample={meta['outer_cv_mean_loglik_per_sample']:.4f} | "
                    f"outer_AIC={meta['outer_cv_mean_aic']:.2f}"
                )
            else:
                print(
                    f"Done mouse {mouse_key} | "
                    f"predicted_rows={meta['n_rows_with_final_predictions']}/{meta['n_rows_original']} | "
                    f"sessions={meta['n_sessions']}"
                )

        except Exception as e:
            df_fail = df_mouse.copy()
            df_fail["Level_2_States"] = np.nan
            df_fail["Level_2_States_OOF"] = np.nan
            for s in range(n_states):
                df_fail[f"Level_2_PostProb_{s}"] = np.nan
                df_fail[f"Level_2_PostProb_OOF_{s}"] = np.nan

            dict_level2[mouse_key] = df_fail
            meta_by_mouse[mouse_key] = {
                "error": str(e),
                "order_col_used": None,
                "n_rows_original": len(df_fail),
                "n_rows_valid_for_modeling": 0,
                "n_rows_with_final_predictions": 0,
                "n_rows_with_oof_predictions": 0,
                "n_sessions": df_fail[session_col_in].nunique() if session_col_in in df_fail.columns else np.nan,
                "outer_cv_table": None,
                "outer_cv_mean_loglik_per_sample": np.nan,
                "outer_cv_sd_loglik_per_sample": np.nan,
                "outer_cv_mean_aic": np.nan,
                "final_model_meta": None,
                "features_used": features,
                "categorical_features_used": categorical_features,
                "use_standardization": use_standardization,
            }
            print(f"Failed mouse {mouse_key}: {e}")

    return dict_level2, meta_by_mouse

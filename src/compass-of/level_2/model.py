"""BGMM-initialized diagonal-covariance GMM-HMM for CoMPASS Level-2.

This module contains model initialization, parameter counting, restart-based
fitting, nested hyperparameter tuning, and time-aware validation helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

try:
    from .level2_preprocessing import (
        fit_preproc_and_transform_train_blocks,
        generate_forward_chaining_splits,
        transform_features,
    )
except ImportError:
    from level2_preprocessing import (
        fit_preproc_and_transform_train_blocks,
        generate_forward_chaining_splits,
        transform_features,
    )

__all__ = [
    "regularize_diag_covars",
    "choose_bgmm_components",
    "fit_bgmm",
    "map_bgmm_components_to_states",
    "make_proto_state_labels_from_bgmm_components",
    "estimate_startprob_and_transmat",
    "fit_state_specific_gmms",
    "initialize_gmmhmm_from_bgmm",
    "fit_one_candidate_with_restarts",
    "choose_best_hyperparams_nested",
]

def regularize_diag_covars(covars: np.ndarray, reg_val: float) -> np.ndarray:
    covars = np.asarray(covars, dtype=float).copy()
    return np.maximum(covars, reg_val)


def choose_bgmm_components(n_states: int, n_mix: int, max_cap: int = 12) -> int:
    return int(min(max_cap, max(n_states * n_mix, n_states + 1)))


def fit_bgmm(
    X_train: np.ndarray,
    n_components: int,
    reg_val: float,
    random_state: int,
) -> BayesianGaussianMixture:
    return BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=reg_val,
        random_state=random_state,
        max_iter=500,
        weight_concentration_prior_type="dirichlet_process",
        init_params="kmeans",
    ).fit(X_train)


def map_bgmm_components_to_states(
    bgmm: BayesianGaussianMixture,
    n_states: int,
    random_state: int,
    min_component_weight: float = 1e-4,
) -> dict[int, int]:
    comp_weights = np.asarray(bgmm.weights_, dtype=float)
    comp_means = np.asarray(bgmm.means_, dtype=float)

    active_idx = np.where(comp_weights > min_component_weight)[0]
    if len(active_idx) == 0:
        active_idx = np.arange(len(comp_weights))

    active_means = comp_means[active_idx]

    if len(active_idx) >= n_states:
        km = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
        active_state_labels = km.fit_predict(active_means)
        comp_to_state = {int(comp): int(st) for comp, st in zip(active_idx, active_state_labels)}
    else:
        comp_to_state = {}
        for i, comp in enumerate(active_idx):
            comp_to_state[int(comp)] = int(i % n_states)

        all_means = comp_means
        active_centers = active_means
        for comp in range(len(comp_weights)):
            if comp not in comp_to_state:
                d = np.sum((active_centers - all_means[comp]) ** 2, axis=1)
                nearest_active = int(np.argmin(d))
                comp_to_state[int(comp)] = comp_to_state[int(active_idx[nearest_active])]

    for comp in range(len(comp_weights)):
        if comp not in comp_to_state:
            d = np.sum((comp_means[active_idx] - comp_means[comp]) ** 2, axis=1)
            nearest_active = int(np.argmin(d))
            comp_to_state[int(comp)] = comp_to_state[int(active_idx[nearest_active])]

    return comp_to_state


def make_proto_state_labels_from_bgmm_components(
    bgmm: BayesianGaussianMixture,
    X_train: np.ndarray,
    comp_to_state: dict[int, int],
) -> np.ndarray:
    comp_labels = bgmm.predict(X_train)
    state_labels = np.array([comp_to_state[int(c)] for c in comp_labels], dtype=int)
    return state_labels


def estimate_startprob_and_transmat(
    proto_labels: np.ndarray,
    lengths: list[int],
    n_states: int,
    pseudo: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    start_counts = np.full(n_states, pseudo, dtype=float)
    trans_counts = np.full((n_states, n_states), pseudo, dtype=float)

    start = 0
    for L in lengths:
        end = start + L
        seq = proto_labels[start:end]

        if len(seq) == 0:
            start = end
            continue

        start_counts[seq[0]] += 1.0

        if len(seq) > 1:
            for i in range(len(seq) - 1):
                trans_counts[seq[i], seq[i + 1]] += 1.0

        start = end

    startprob = start_counts / start_counts.sum()
    transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return startprob, transmat


def fit_state_specific_gmms(
    X_train: np.ndarray,
    proto_labels: np.ndarray,
    n_states: int,
    n_mix: int,
    reg_val: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]

    means = np.zeros((n_states, n_mix, n_features), dtype=float)
    covars = np.zeros((n_states, n_mix, n_features), dtype=float)
    weights = np.zeros((n_states, n_mix), dtype=float)

    global_gmm = GaussianMixture(
        n_components=n_mix,
        covariance_type="diag",
        reg_covar=reg_val,
        random_state=random_state,
        max_iter=500,
        n_init=3,
        init_params="kmeans",
    ).fit(X_train)

    for s in range(n_states):
        idx = np.where(proto_labels == s)[0]

        if len(idx) < max(10 * n_mix, n_mix + 2):
            gmm = global_gmm
        else:
            try:
                gmm = GaussianMixture(
                    n_components=n_mix,
                    covariance_type="diag",
                    reg_covar=reg_val,
                    random_state=random_state + s,
                    max_iter=500,
                    n_init=3,
                    init_params="kmeans",
                ).fit(X_train[idx])
            except Exception:
                gmm = global_gmm

        means[s] = gmm.means_
        covars[s] = regularize_diag_covars(gmm.covariances_, reg_val=reg_val)
        weights[s] = gmm.weights_ / gmm.weights_.sum()

    return means, covars, weights


def initialize_gmmhmm_from_bgmm(
    X_train: np.ndarray,
    lengths_train: list[int],
    n_states: int,
    n_mix: int,
    reg_val: float,
    random_state: int,
    n_iter: int = 400,
) -> GMMHMM:
    n_bgmm_components = choose_bgmm_components(n_states=n_states, n_mix=n_mix)

    bgmm = fit_bgmm(
        X_train=X_train,
        n_components=n_bgmm_components,
        reg_val=reg_val,
        random_state=random_state,
    )

    comp_to_state = map_bgmm_components_to_states(
        bgmm=bgmm,
        n_states=n_states,
        random_state=random_state,
    )

    proto_labels = make_proto_state_labels_from_bgmm_components(
        bgmm=bgmm,
        X_train=X_train,
        comp_to_state=comp_to_state,
    )

    startprob, transmat = estimate_startprob_and_transmat(
        proto_labels=proto_labels,
        lengths=lengths_train,
        n_states=n_states,
        pseudo=1e-2,
    )

    means, covars, weights = fit_state_specific_gmms(
        X_train=X_train,
        proto_labels=proto_labels,
        n_states=n_states,
        n_mix=n_mix,
        reg_val=reg_val,
        random_state=random_state,
    )

    model = GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type="diag",
        random_state=random_state,
        n_iter=n_iter,
        tol=1e-3,
        init_params="",
        params="stmcw",
        min_covar=reg_val,
        verbose=False,
    )

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    model.weights_ = weights

    return model


def _estimate_num_params_gmmhmm(
    model: GMMHMM,
    n_features: int,
) -> int:
    """
    Approximate parameter count for diag-covariance GMMHMM.
    Good enough for AIC comparison within this workflow.
    """
    n_states = int(model.n_components)
    n_mix = int(model.n_mix)

    startprob_params = n_states - 1
    transmat_params = n_states * (n_states - 1)
    mixture_weight_params = n_states * (n_mix - 1)
    mean_params = n_states * n_mix * n_features
    covar_params = n_states * n_mix * n_features

    return (
        startprob_params
        + transmat_params
        + mixture_weight_params
        + mean_params
        + covar_params
    )


def _compute_aic_from_loglik(
    loglik_total: float,
    model: GMMHMM,
    n_features: int,
) -> float:
    k = _estimate_num_params_gmmhmm(model=model, n_features=n_features)
    return float(2 * k - 2 * loglik_total)


def fit_one_candidate_with_restarts(
    X_train: np.ndarray,
    lengths_train: list[int],
    n_states: int,
    n_mix: int,
    reg_val: float,
    restart_seeds: list[int],
    n_iter: int = 400,
    verbose: bool = False,
) -> tuple[GMMHMM, float]:
    best_model = None
    best_train_loglik = -np.inf

    for seed in restart_seeds:
        try:
            model = initialize_gmmhmm_from_bgmm(
                X_train=X_train,
                lengths_train=lengths_train,
                n_states=n_states,
                n_mix=n_mix,
                reg_val=reg_val,
                random_state=seed,
                n_iter=n_iter,
            )
            model.fit(X_train, lengths=lengths_train)
            train_ll = model.score(X_train, lengths=lengths_train)

            if np.isfinite(train_ll) and train_ll > best_train_loglik:
                best_train_loglik = train_ll
                best_model = model

        except Exception as e:
            if verbose:
                print(f"[restart failed] seed={seed} mix={n_mix} reg={reg_val:.0e} | {e}")

    if best_model is None:
        raise ValueError(
            f"All restarts failed for n_states={n_states}, n_mix={n_mix}, reg={reg_val:.0e}"
        )

    return best_model, best_train_loglik


# ============================================================
# Inner tuning
# ============================================================

def _score_candidate_on_session_split(
    train_blocks: list[pd.DataFrame],
    val_blocks: list[pd.DataFrame],
    features: list[str],
    categorical_features: list[str] | None,
    use_standardization: bool,
    n_states: int,
    n_mix: int,
    reg_val: float,
    restart_seeds: list[int],
    n_iter: int,
    verbose: bool = False,
) -> float:
    preproc, X_train, lengths_train = fit_preproc_and_transform_train_blocks(
        train_blocks=train_blocks,
        features=features,
        categorical_features=categorical_features,
        use_standardization=use_standardization,
    )

    X_val_blocks = []
    val_lengths = []
    for blk in val_blocks:
        if len(blk) == 0:
            continue
        X_blk = transform_features(blk, preproc)
        X_val_blocks.append(X_blk)
        val_lengths.append(len(X_blk))

    if len(X_val_blocks) == 0:
        raise ValueError("No validation blocks available.")

    X_val = np.vstack(X_val_blocks)

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

    val_ll_total = model.score(X_val, lengths=val_lengths)
    return float(val_ll_total / max(len(X_val), 1))


def choose_best_hyperparams_nested(
    session_dict_train: dict[Any, pd.DataFrame],
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
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    if n_mix_options is None:
        n_mix_options = [1, 2]
    if reg_options is None:
        reg_options = [1e-3, 1e-4]
    if n_splits_options is None:
        n_splits_options = [3, 4]
    if restart_seeds is None:
        restart_seeds = [3, 7, 11, 19, 23]

    train_sessions = list(session_dict_train.keys())
    n_train_sessions = len(train_sessions)

    results = []
    best_params = None
    best_score = -np.inf

    def update_best(row: dict):
        nonlocal best_params, best_score
        mean_score = row["mean_val_loglik_per_sample"]

        is_better = False
        if mean_score > best_score:
            is_better = True
        elif np.isclose(mean_score, best_score):
            if best_params is None:
                is_better = True
            else:
                old_complexity = (best_params["n_mix"], -best_params["reg_val"])
                new_complexity = (row["n_mix"], -row["reg_val"])
                if new_complexity < old_complexity:
                    is_better = True

        if is_better:
            best_score = mean_score
            best_params = {
                "n_states": row["n_states"],
                "n_mix": row["n_mix"],
                "reg_val": row["reg_val"],
                "inner_strategy": row["inner_strategy"],
            }

    if n_train_sessions >= 3:
        for n_mix in n_mix_options:
            for reg_val in reg_options:
                fold_scores = []

                for heldout_inner in train_sessions:
                    inner_train_keys = [s for s in train_sessions if s != heldout_inner]
                    train_blocks = [session_dict_train[s].copy() for s in inner_train_keys]
                    val_blocks = [session_dict_train[heldout_inner].copy()]

                    try:
                        score = _score_candidate_on_session_split(
                            train_blocks=train_blocks,
                            val_blocks=val_blocks,
                            features=features,
                            categorical_features=categorical_features,
                            use_standardization=use_standardization,
                            n_states=n_states,
                            n_mix=n_mix,
                            reg_val=reg_val,
                            restart_seeds=restart_seeds,
                            n_iter=n_iter,
                            verbose=verbose,
                        )
                        if np.isfinite(score):
                            fold_scores.append(score)
                    except Exception as e:
                        if verbose:
                            print(f"[inner LOSO failed] {heldout_inner} mix={n_mix} reg={reg_val:.0e} | {e}")

                if len(fold_scores) > 0:
                    row = {
                        "n_states": n_states,
                        "n_mix": n_mix,
                        "reg_val": reg_val,
                        "mean_val_loglik_per_sample": float(np.mean(fold_scores)),
                        "std_val_loglik_per_sample": float(np.std(fold_scores, ddof=0)),
                        "n_successful_folds": int(len(fold_scores)),
                        "inner_strategy": "session_loso",
                    }
                    results.append(row)
                    update_best(row)

    if best_params is None:
        for n_mix in n_mix_options:
            for reg_val in reg_options:
                for n_splits in n_splits_options:
                    fold_scores = []

                    for target_sess in train_sessions:
                        df_target = session_dict_train[target_sess].copy()
                        splits = generate_forward_chaining_splits(
                            n_rows=len(df_target),
                            n_splits=n_splits,
                            min_train_size=min_train_size,
                            min_val_size=min_val_size,
                            gap=gap,
                        )
                        if len(splits) == 0:
                            continue

                        other_sessions = [session_dict_train[s].copy() for s in train_sessions if s != target_sess]

                        for tr_idx, va_idx in splits:
                            df_target_train = df_target.iloc[tr_idx].copy()
                            df_target_val = df_target.iloc[va_idx].copy()

                            train_blocks = other_sessions + [df_target_train]
                            val_blocks = [df_target_val]

                            try:
                                score = _score_candidate_on_session_split(
                                    train_blocks=train_blocks,
                                    val_blocks=val_blocks,
                                    features=features,
                                    categorical_features=categorical_features,
                                    use_standardization=use_standardization,
                                    n_states=n_states,
                                    n_mix=n_mix,
                                    reg_val=reg_val,
                                    restart_seeds=restart_seeds,
                                    n_iter=n_iter,
                                    verbose=verbose,
                                )
                                if np.isfinite(score):
                                    fold_scores.append(score)
                            except Exception as e:
                                if verbose:
                                    print(f"[inner forward failed] {target_sess} mix={n_mix} reg={reg_val:.0e} | {e}")

                    if len(fold_scores) > 0:
                        row = {
                            "n_states": n_states,
                            "n_mix": n_mix,
                            "reg_val": reg_val,
                            "mean_val_loglik_per_sample": float(np.mean(fold_scores)),
                            "std_val_loglik_per_sample": float(np.std(fold_scores, ddof=0)),
                            "n_successful_folds": int(len(fold_scores)),
                            "inner_strategy": f"forward_chaining_{n_splits}",
                        }
                        results.append(row)
                        update_best(row)

    if best_params is None:
        best_params = {
            "n_states": n_states,
            "n_mix": min(n_mix_options),
            "reg_val": max(reg_options),
            "inner_strategy": "default_fallback",
        }
        results.append({
            "n_states": n_states,
            "n_mix": min(n_mix_options),
            "reg_val": max(reg_options),
            "mean_val_loglik_per_sample": np.nan,
            "std_val_loglik_per_sample": np.nan,
            "n_successful_folds": 0,
            "inner_strategy": "default_fallback",
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["mean_val_loglik_per_sample", "std_val_loglik_per_sample", "n_mix", "reg_val"],
            ascending=[False, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    return best_params, results_df

"""Level-1 Gamma(step) + von Mises(angle) hidden Markov model."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from scipy.special import gammaln, i0e, i1e
from scipy.optimize import minimize
from tqdm.auto import tqdm


# =========================================================
# Basic helpers
# =========================================================
def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _contig_lengths(ids):
    lengths = []
    last = None
    c = 0
    for v in ids:
        if last is None or v == last:
            c += 1
        else:
            lengths.append(c)
            c = 1
        last = v
    if c > 0:
        lengths.append(c)
    return lengths


def _stationary_dist(A):
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = np.maximum(pi, 0)
    if pi.sum() <= 0:
        pi = np.ones(A.shape[0], dtype=float)
    pi = pi / pi.sum()
    return pi


def _aic_from_loglik(loglik, n_states):
    # start probs: S-1
    # trans probs: S*(S-1)
    # emissions/state: Gamma(step)=2, VM(angle)=2
    k = (n_states - 1) + n_states * (n_states - 1) + 4 * n_states
    return 2 * k - 2 * loglik


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    x = x[m]
    w = w[m]
    if x.size == 0 or np.sum(w) <= 0:
        return np.nan
    w = w / np.sum(w)
    return float(np.sum(w * x))


# =========================================================
# Circular summaries
# =========================================================
def _vm_resultant_length_from_kappa(kappa):
    kappa = np.asarray(kappa, dtype=float)
    return np.exp(np.log(i1e(kappa)) - np.log(i0e(kappa)))


def _circ_mean_weighted(angle, w):
    angle = np.asarray(angle, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(angle) & np.isfinite(w)
    angle = angle[m]
    w = w[m]
    if angle.size == 0 or np.sum(w) <= 0:
        return np.nan

    w = w / np.sum(w)
    s = np.sum(w * np.sin(angle))
    c = np.sum(w * np.cos(angle))
    return float(np.arctan2(s, c))


def _circ_resultant_length_weighted(angle, w):
    angle = np.asarray(angle, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(angle) & np.isfinite(w)
    angle = angle[m]
    w = w[m]
    if angle.size == 0 or np.sum(w) <= 0:
        return np.nan

    w = w / np.sum(w)
    s = np.sum(w * np.sin(angle))
    c = np.sum(w * np.cos(angle))
    return float(np.sqrt(s**2 + c**2))


def _circ_sd_from_R(R):
    if not np.isfinite(R) or R <= 0:
        return np.nan
    return float(np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0))))


# =========================================================
# Parameter helpers
# =========================================================
def _gamma_ktheta_from_mean_sd(mean_val, sd_val):
    mean_val = max(float(mean_val), 1e-8)
    sd_val = max(float(sd_val), 1e-8)
    k = (mean_val / sd_val) ** 2
    theta = (sd_val**2) / mean_val
    return k, theta


def _gamma_ktheta_bounds(mean_range, sd_range):
    m_lo, m_hi = mean_range
    s_lo, s_hi = sd_range

    m_lo = max(m_lo, 1e-6)
    m_hi = max(m_hi, m_lo * 1.001)
    s_lo = max(s_lo, 1e-6)
    s_hi = max(s_hi, s_lo * 1.001)

    corners = [
        _gamma_ktheta_from_mean_sd(m_lo, s_lo),
        _gamma_ktheta_from_mean_sd(m_lo, s_hi),
        _gamma_ktheta_from_mean_sd(m_hi, s_lo),
        _gamma_ktheta_from_mean_sd(m_hi, s_hi),
    ]
    ks = [c[0] for c in corners]
    ths = [c[1] for c in corners]

    k_bounds = (max(1e-4, float(min(ks)) * 0.5), float(max(ks)) * 2.0)
    th_bounds = (max(1e-6, float(min(ths)) * 0.5), float(max(ths)) * 2.0)
    return k_bounds, th_bounds


def compute_parameter_ranges_vm(
    df: pd.DataFrame,
    step_col: str = "step",
    angle_col: str = "angle",
) -> Dict[str, Tuple[float, float]]:
    step = df[step_col].to_numpy(float)
    step = step[np.isfinite(step)]

    angle = df[angle_col].to_numpy(float)
    angle = angle[np.isfinite(angle)]
    angle = _wrap_angle(angle)

    step_iqr = _iqr(step)
    step_med = np.nanmedian(step) if step.size else 1.0

    step_mean_range = (
        max(1e-6, step_med - step_iqr),
        max(step_med + step_iqr, step_med * 1.5, 1e-3),
    )
    step_sd_range = (
        0.1,
        max(0.2, 1.5 * _mad(step)),
    )

    c = np.nanmean(np.cos(angle)) if angle.size else 0.0
    s = np.nanmean(np.sin(angle)) if angle.size else 0.0
    R = np.sqrt(c**2 + s**2) if np.isfinite(c) and np.isfinite(s) else 0.0

    if R > 0:
        if R < 0.53:
            kappa = 2 * R + R**3 + (5 * R**5) / 6
        elif R < 0.85:
            kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            denom = (R**3 - 4 * R**2 + 3 * R)
            kappa = 1 / denom if denom != 0 else 10.0
        kappa_hi = float(min(max(5.0, 2 * kappa + 1.0), 60.0))
        angle_conc_range = (0.05, kappa_hi)
    else:
        angle_conc_range = (0.05, 20.0)

    return {
        "step_mean_range": step_mean_range,
        "step_sd_range": step_sd_range,
        "angle_conc_range": angle_conc_range,
    }


# =========================================================
# Log densities
# =========================================================
def logpdf_gamma(x, k, theta):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    k = np.clip(k, 1e-8, None)
    theta = np.clip(theta, 1e-8, None)
    return (k - 1) * np.log(x) - (x / theta) - k * np.log(theta) - gammaln(k)


def logpdf_vonmises(phi, mu, kappa):
    kappa = np.clip(kappa, 0.0, None)
    logC = -(np.log(2 * np.pi) + (np.log(i0e(kappa)) + np.abs(kappa)))
    return kappa * np.cos(phi - mu) + logC


# =========================================================
# Forward-backward / Viterbi
# =========================================================
def forward_backward(log_lik, startprob, transmat, lengths):
    S = startprob.shape[0]
    N = log_lik.shape[0]

    log_start = np.log(startprob + 1e-15)
    log_A = np.log(transmat + 1e-15)

    alpha = np.zeros((N, S), dtype=float)
    scales = np.zeros(N, dtype=float)

    idx = 0
    for L in lengths:
        a = log_start + log_lik[idx]
        c = np.logaddexp.reduce(a)
        alpha[idx] = a - c
        scales[idx] = c

        for t in range(idx + 1, idx + L):
            a = np.logaddexp.reduce(alpha[t - 1][:, None] + log_A, axis=0) + log_lik[t]
            c = np.logaddexp.reduce(a)
            alpha[t] = a - c
            scales[t] = c
        idx += L

    beta = np.zeros((N, S), dtype=float)
    idx = 0
    for L in lengths:
        end = idx + L - 1
        beta[end] = 0.0
        for t in range(end - 1, idx - 1, -1):
            b = np.logaddexp.reduce(
                log_A + (log_lik[t + 1] + beta[t + 1])[None, :],
                axis=1,
            )
            beta[t] = b - scales[t + 1]
        idx += L

    log_gamma = alpha + beta
    log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)

    xisum = np.zeros_like(transmat, dtype=float)
    idx = 0
    for L in lengths:
        for t in range(idx, idx + L - 1):
            M = alpha[t][:, None] + log_A + log_lik[t + 1][None, :] + beta[t + 1][None, :]
            M -= np.max(M)
            P = np.exp(M)
            P /= P.sum()
            xisum += P
        idx += L

    total_ll = float(scales.sum())
    return gamma, xisum, total_ll


def viterbi(log_lik, startprob, transmat, lengths):
    S = startprob.shape[0]
    N = log_lik.shape[0]
    path = np.empty(N, dtype=int)

    log_start = np.log(startprob + 1e-15)
    log_A = np.log(transmat + 1e-15)

    idx = 0
    for L in lengths:
        delta = log_start + log_lik[idx]
        psi = np.zeros((L, S), dtype=int)
        deltas = np.zeros((L, S), dtype=float)
        deltas[0] = delta

        for t in range(1, L):
            prev = deltas[t - 1][:, None] + log_A
            psi[t] = np.argmax(prev, axis=0)
            deltas[t] = np.max(prev, axis=0) + log_lik[idx + t]

        sT = int(np.argmax(deltas[L - 1]))
        seq = np.empty(L, dtype=int)
        seq[L - 1] = sT

        for t in range(L - 2, -1, -1):
            seq[t] = psi[t + 1, seq[t + 1]]

        path[idx : idx + L] = seq
        idx += L

    return path


# =========================================================
# More momentuHMM-like initialization
# =========================================================
def _initial_step_params_from_quantiles(step, n_states):
    step = np.asarray(step, dtype=float)
    step = step[np.isfinite(step)]

    if step.size == 0:
        return np.ones(n_states), np.ones(n_states)

    probs = np.linspace(0.25, 0.75, n_states)
    step_means = np.quantile(step, probs)

    global_sd = max(_mad(step), np.std(step) * 0.5, 0.1)

    k_list = []
    th_list = []
    for m in step_means:
        k, th = _gamma_ktheta_from_mean_sd(max(m, 1e-4), global_sd)
        k_list.append(k)
        th_list.append(th)

    return np.array(k_list, dtype=float), np.array(th_list, dtype=float)


def _initial_vm_params(angle, n_states, rng, kappa_bounds=(0.05, 20.0)):
    """
    momentuHMM-like practical init:
    - mu near 0 for all states
    - states separated mainly by concentration (kappa), not radically different means
    """
    lo, hi = kappa_bounds

    mu0 = np.zeros(n_states, dtype=float)
    mu0 = _wrap_angle(mu0 + rng.normal(0, 0.03, size=n_states))

    if n_states == 1:
        kappas = np.array([min(max(2.0, lo), hi)], dtype=float)
    elif n_states == 2:
        kappas = np.array([max(lo, 0.5), min(hi, 5.0)], dtype=float)
    else:
        kappas = np.linspace(max(lo, 0.5), min(hi, 8.0), n_states)

    kappas = np.clip(kappas + rng.normal(0, 0.05, size=n_states), lo, hi)
    return mu0, kappas


# =========================================================
# HMM class: Gamma(step) + von Mises(angle)
# =========================================================
class GammaVonMisesHMM:
    def __init__(
        self,
        n_states: int = 2,
        max_iter: int = 200,
        tol: float = 1e-4,
        method: str = "L-BFGS-B",
        random_state: int = 0,
        sticky_init: bool = True,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        use_bounds: bool = True,
    ):
        self.S = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.rng = np.random.default_rng(random_state)
        self.sticky_init = sticky_init
        self.param_ranges = param_ranges
        self.use_bounds = use_bounds

    def _init_params(self, step, angle):
        S = self.S
        self.startprob_ = np.full(S, 1.0 / S)

        diag = 0.95 if self.sticky_init else 0.90
        self.transmat_ = np.full((S, S), (1 - diag) / max(S - 1, 1))
        np.fill_diagonal(self.transmat_, diag)

        self.k_step_, self.theta_step_ = _initial_step_params_from_quantiles(step, S)

        if self.param_ranges is not None and self.param_ranges.get("angle_conc_range") is not None:
            kb = self.param_ranges["angle_conc_range"]
        else:
            kb = (0.05, 20.0)

        self.mu_, self.kappa_ = _initial_vm_params(angle, S, self.rng, kappa_bounds=kb)

    def _loglik(self, step, angle):
        N = len(step)
        L = np.zeros((N, self.S), dtype=float)
        for s in range(self.S):
            L[:, s] = (
                logpdf_gamma(step, self.k_step_[s], self.theta_step_[s])
                + logpdf_vonmises(angle, self.mu_[s], self.kappa_[s])
            )
        return L

    def _bounds_for_state(self):
        if (not self.use_bounds) or (self.param_ranges is None):
            return None

        sm = self.param_ranges["step_mean_range"]
        ss = self.param_ranges["step_sd_range"]
        k_bounds_s, th_bounds_s = _gamma_ktheta_bounds(sm, ss)

        kc = self.param_ranges.get("angle_conc_range") or (0.05, 20.0)

        b_logk_s = (np.log(k_bounds_s[0]), np.log(k_bounds_s[1]))
        b_logth_s = (np.log(th_bounds_s[0]), np.log(th_bounds_s[1]))
        b_mu = (-np.pi, np.pi)
        b_logkp1 = (np.log(kc[0] + 1.0), np.log(kc[1] + 1.0))
        return [b_logk_s, b_logth_s, b_mu, b_logkp1]

    def _mstep_emissions(self, step, angle, gamma):
        bounds = self._bounds_for_state()

        for s in range(self.S):
            w = gamma[:, s]
            w = w / (w.sum() + 1e-12)

            def nll(p):
                logk_s, logth_s, mu, logkp1 = p
                k_s = np.exp(logk_s)
                th_s = np.exp(logth_s)
                kp = np.exp(logkp1) - 1.0
                return -(w * (logpdf_gamma(step, k_s, th_s) + logpdf_vonmises(angle, mu, kp))).sum()

            x0 = np.array([
                np.log(self.k_step_[s]),
                np.log(self.theta_step_[s]),
                self.mu_[s],
                np.log(self.kappa_[s] + 1.0),
            ])

            res = minimize(
                nll,
                x0,
                method=self.method,
                bounds=bounds,
                options=dict(maxiter=500, disp=False),
            )

            if res.success:
                logk_s, logth_s, mu, logkp1 = res.x
                self.k_step_[s] = np.exp(logk_s)
                self.theta_step_[s] = np.exp(logth_s)
                self.mu_[s] = _wrap_angle(mu)
                self.kappa_[s] = np.exp(logkp1) - 1.0

    def fit(self, df, seq_col="Session", step_col="step", angle_col="angle"):
        ids = df[seq_col].to_numpy()
        step = df[step_col].to_numpy(float)
        ang = _wrap_angle(df[angle_col].to_numpy(float))

        mask = np.isfinite(step) & np.isfinite(ang)
        ids = ids[mask]
        step = step[mask]
        ang = ang[mask]

        lengths = _contig_lengths(ids)

        self._fit_ids_ = ids.copy()
        self._fit_step_ = step.copy()
        self._fit_angle_ = ang.copy()
        self.lengths_ = lengths

        self._init_params(step, ang)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            logL = self._loglik(step, ang)
            gamma, xisum, ll = forward_backward(logL, self.startprob_, self.transmat_, lengths)

            start = np.zeros(self.S, dtype=float)
            idx = 0
            for L in lengths:
                start += gamma[idx]
                idx += L
            self.startprob_ = (start / len(lengths)).clip(1e-12)
            self.startprob_ /= self.startprob_.sum()

            A = xisum.clip(1e-12)
            self.transmat_ = A / A.sum(axis=1, keepdims=True)

            self._mstep_emissions(step, ang, gamma)

            if ll - prev_ll < self.tol:
                break
            prev_ll = ll

        self.posterior_ = gamma
        self.states_post_ = gamma.argmax(axis=1)
        self.viterbi_ = viterbi(self._loglik(step, ang), self.startprob_, self.transmat_, lengths)
        self.loglik_ = float(ll)
        self.AIC_ = _aic_from_loglik(self.loglik_, self.S)

        # probability summaries
        self.stationary_ = _stationary_dist(self.transmat_)
        self.occupancy_ = self.posterior_.mean(axis=0)

        # state-wise summaries
        step_means = np.zeros(self.S, dtype=float)
        angle_circ_mean = np.zeros(self.S, dtype=float)
        angle_R = np.zeros(self.S, dtype=float)
        angle_circ_sd = np.zeros(self.S, dtype=float)

        for s in range(self.S):
            w = self.posterior_[:, s]
            step_means[s] = _weighted_mean(step, w)
            angle_circ_mean[s] = _circ_mean_weighted(ang, w)
            angle_R[s] = _circ_resultant_length_weighted(ang, w)
            angle_circ_sd[s] = _circ_sd_from_R(angle_R[s])

        self.step_means_ = step_means
        self.angle_circ_mean_ = angle_circ_mean
        self.angle_R_ = angle_R
        self.angle_circ_sd_ = angle_circ_sd
        self.vm_R_from_kappa_ = _vm_resultant_length_from_kappa(self.kappa_)

        return self

    def score(self, df, seq_col="Session", step_col="step", angle_col="angle"):
        ids = df[seq_col].to_numpy()
        step = df[step_col].to_numpy(float)
        ang = _wrap_angle(df[angle_col].to_numpy(float))

        m = np.isfinite(step) & np.isfinite(ang)
        ids = ids[m]
        step = step[m]
        ang = ang[m]

        lengths = _contig_lengths(ids)

        _, _, ll = forward_backward(
            self._loglik(step, ang),
            self.startprob_,
            self.transmat_,
            lengths,
        )
        return float(ll)

    def reorder_states(self, order: List[int]):
        order = np.asarray(order, dtype=int)

        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[order][:, order]
        self.k_step_ = self.k_step_[order]
        self.theta_step_ = self.theta_step_[order]
        self.mu_ = self.mu_[order]
        self.kappa_ = self.kappa_[order]

        if hasattr(self, "posterior_"):
            self.posterior_ = self.posterior_[:, order]
            self.states_post_ = np.argmax(self.posterior_, axis=1)

        if hasattr(self, "viterbi_"):
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            self.viterbi_ = inv[self.viterbi_]

        if hasattr(self, "stationary_"):
            self.stationary_ = self.stationary_[order]
        if hasattr(self, "occupancy_"):
            self.occupancy_ = self.occupancy_[order]
        if hasattr(self, "step_means_"):
            self.step_means_ = self.step_means_[order]
        if hasattr(self, "angle_circ_mean_"):
            self.angle_circ_mean_ = self.angle_circ_mean_[order]
        if hasattr(self, "angle_R_"):
            self.angle_R_ = self.angle_R_[order]
        if hasattr(self, "angle_circ_sd_"):
            self.angle_circ_sd_ = self.angle_circ_sd_[order]
        if hasattr(self, "vm_R_from_kappa_"):
            self.vm_R_from_kappa_ = self.vm_R_from_kappa_[order]


# =========================================================
# Optional post-fit reordering
# =========================================================
def reorder_lowstep_highturn_first(model: GammaVonMisesHMM):
    # lower step and higher turn variability first
    step = model.step_means_
    turn_var = model.angle_circ_sd_

    score = (-(step - step.mean()) / (step.std() + 1e-12)
             + (turn_var - turn_var.mean()) / (turn_var.std() + 1e-12))

    order = np.argsort(-score)
    model.reorder_states(order.tolist())
    return model


# =========================================================
# Summaries
# =========================================================
def summarize_hmm(model: GammaVonMisesHMM) -> Dict[str, Any]:
    return {
        "n_states": model.S,
        "optimizer": model.method,
        "loglik": float(model.loglik_),
        "AIC": float(model.AIC_),
        "startprob": np.round(model.startprob_, 4).tolist(),
        "stationary_prob": np.round(model.stationary_, 4).tolist(),
        "occupancy_prob": np.round(model.occupancy_, 4).tolist(),
        "transmat": np.round(model.transmat_, 4).tolist(),
        "step_k": np.round(model.k_step_, 4).tolist(),
        "step_theta": np.round(model.theta_step_, 4).tolist(),
        "step_mean_post": np.round(model.step_means_, 4).tolist(),
        "vm_mu": np.round(model.mu_, 4).tolist(),
        "vm_kappa": np.round(model.kappa_, 4).tolist(),
        "vm_resultant_length_from_kappa": np.round(model.vm_R_from_kappa_, 4).tolist(),
        "angle_circ_mean_post": np.round(model.angle_circ_mean_, 4).tolist(),
        "angle_R_post": np.round(model.angle_R_, 4).tolist(),
        "angle_circ_sd_post": np.round(model.angle_circ_sd_, 4).tolist(),
    }


def print_hmm_summary(summary: Dict[str, Any]):
    print("\nBest Model Characteristics:")
    print(f"Optimizer: {summary['optimizer']}")
    print(f"Number of states: {summary['n_states']}")
    print(f"AIC: {summary['AIC']:.2f} | logLik: {summary['loglik']:.2f}")
    print(f"Start probabilities: {summary['startprob']}")
    print(f"Stationary probabilities: {summary['stationary_prob']}")
    print(f"Posterior occupancy probabilities: {summary['occupancy_prob']}")
    print("Transition matrix:")
    print(np.array(summary["transmat"]))
    print(f"Step k: {summary['step_k']}")
    print(f"Step theta: {summary['step_theta']}")
    print(f"Posterior-weighted step mean: {summary['step_mean_post']}")
    print(f"VM mu: {summary['vm_mu']}")
    print(f"VM kappa: {summary['vm_kappa']}")
    print(f"VM resultant length from kappa: {summary['vm_resultant_length_from_kappa']}")
    print(f"Posterior circular mean of angle: {summary['angle_circ_mean_post']}")
    print(f"Posterior resultant length R: {summary['angle_R_post']}")
    print(f"Posterior circular SD: {summary['angle_circ_sd_post']}\n")


# =========================================================
# Sequence/session-wise posterior summaries
# =========================================================
def compute_sequence_state_probabilities(
    fitted_df: pd.DataFrame,
    seq_col: str = "Session",
    state_col: str = "HMM_State",
    posterior_prefix: str = "Post_Prob_",
    n_states: Optional[int] = None,
) -> pd.DataFrame:
    out = []

    if n_states is None:
        probs = [c for c in fitted_df.columns if c.startswith(posterior_prefix)]
        n_states = len(probs)

    for seq_val, g in fitted_df.groupby(seq_col, sort=False):
        row = {seq_col: seq_val, "n_obs": len(g)}

        hard = g[state_col].dropna() if state_col in g.columns else pd.Series(dtype=float)
        for s in range(1, n_states + 1):
            row[f"hard_state{s}_prop"] = float(np.mean(hard == s)) if len(hard) else np.nan

        for s in range(1, n_states + 1):
            pcol = f"{posterior_prefix}{s}"
            row[f"post_state{s}_mean"] = float(g[pcol].mean(skipna=True)) if pcol in g.columns else np.nan

        out.append(row)

    return pd.DataFrame(out)


# =========================================================
# Best result container
# =========================================================
@dataclass
class BestResult:
    model: GammaVonMisesHMM
    summary: Dict[str, Any]
    records: pd.DataFrame
    data: pd.DataFrame
    sequence_probs: pd.DataFrame


# =========================================================
# Main orchestrator
# =========================================================
def fit_best_hmm_momentu_style(
    preproc_df: pd.DataFrame,
    n_states: int = 2,
    n_repetitions: int = 20,
    opt_methods: List[str] = ["L-BFGS-B", "BFGS", "Powell"],
    max_iter: int = 200,
    sticky_init: bool = True,
    use_data_driven_ranges: bool = True,
    seq_col: str = "Session",
    step_col: str = "step",
    angle_col: str = "angle",
    seed: int = 123,
    reorder_states_after_fit: bool = True,
    show_progress: bool = True,
) -> BestResult:
    rng = np.random.default_rng(seed)
    base = preproc_df.copy()

    records = []
    candidates: List[Tuple[GammaVonMisesHMM, pd.DataFrame, Dict[str, Any]]] = []

    total = len(opt_methods) * n_repetitions
    pbar = tqdm(total=total, desc="Gamma-VM HMM search", leave=False) if show_progress else None

    ranges = None
    if use_data_driven_ranges:
        ranges = compute_parameter_ranges_vm(base, step_col=step_col, angle_col=angle_col)

    for opt in opt_methods:
        for _ in range(n_repetitions):
            this_seed = int(rng.integers(0, 10_000_000))
            try:
                model = GammaVonMisesHMM(
                    n_states=n_states,
                    max_iter=max_iter,
                    tol=1e-4,
                    method=opt,
                    random_state=this_seed,
                    sticky_init=sticky_init,
                    param_ranges=ranges,
                    use_bounds=True,
                )

                model.fit(base, seq_col=seq_col, step_col=step_col, angle_col=angle_col)

                if reorder_states_after_fit:
                    model = reorder_lowstep_highturn_first(model)

                summ = summarize_hmm(model)

                records.append({
                    "optimizer": opt,
                    "seed": this_seed,
                    "AIC": summ["AIC"],
                    "loglik": summ["loglik"],
                })

                out = base.copy()
                out["HMM_State"] = np.nan

                for s in range(model.S):
                    out[f"Post_Prob_{s+1}"] = np.nan

                mask = np.isfinite(base[step_col].to_numpy(float)) & np.isfinite(base[angle_col].to_numpy(float))

                out.loc[mask, "HMM_State"] = (model.viterbi_ + 1).astype(int)

                # print/store posterior state probabilities row-wise
                for s in range(model.S):
                    out.loc[mask, f"Post_Prob_{s+1}"] = model.posterior_[:, s]

                candidates.append((model, out, summ))

            except Exception:
                pass
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    if len(candidates) == 0:
        raise RuntimeError("No valid models converged.")

    best_idx = int(np.argmin([c[2]["AIC"] for c in candidates]))
    best_model, df_with_states, best_summary = candidates[best_idx]

    rec_df = pd.DataFrame.from_records(records)
    if not rec_df.empty:
        rec_df = rec_df.sort_values(
            ["AIC", "optimizer"],
            ascending=[True, True],
            kind="mergesort",
        )

    seq_probs = compute_sequence_state_probabilities(
        fitted_df=df_with_states,
        seq_col=seq_col,
        state_col="HMM_State",
        posterior_prefix="Post_Prob_",
        n_states=best_model.S,
    )

    return BestResult(
        model=best_model,
        summary=best_summary,
        records=rec_df,
        data=df_with_states,
        sequence_probs=seq_probs,
    )
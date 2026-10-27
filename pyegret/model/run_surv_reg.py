"""
Censored (Tobit) log-normal regression at a single WRTDS grid point.

Mirrors R EGRET's ``runSurvReg()``, which fits ``survival::survreg()`` with a
Gaussian error distribution to interval-censored log-concentrations
(``Surv(log(ConcLow), log(ConcHigh), type = "interval2")``), weighted by the
tricube kernel. Ordinary non-detects (ConcLow = 0) are the left-censored
special case.

Conditioning strategy
---------------------
The design matrix is *centered on the target point*::

    X_c = [1, DecYear - yr, LogQ - lq, SinDY - s, CosDY - c]

so every predictor column is O(1) regardless of the absolute year, and the
intercept *is* the fitted value at the target (``x_target_c = [1,0,0,0,0]``).
Without centering, ``DecYear ~ 2000`` and ``beta_0 ~ -2000 * beta_time``
create a near-flat ridge that no optimiser handles gracefully.

Optimiser
---------
The censored Gaussian log-likelihood is smooth and its gradient is available
in closed form, so this uses gradient-based L-BFGS-B rather than a
derivative-free simplex.  On a typical grid point that is roughly an order of
magnitude faster than Nelder-Mead, which matters because the corrected
surface grid holds thousands of points rather than a few hundred.
Nelder-Mead remains as a fallback for the rare case where L-BFGS-B fails to
return a finite result.

The weights are *case weights* on the log-likelihood, exactly as
``survreg(weights=)`` treats them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr

__all__ = ["run_surv_reg"]

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)

# Bounds are deliberately generous: they exist to keep the optimiser out of
# numerically degenerate regions, not to constrain the science. sigma ranges
# over exp([-8, 4]) = [3.4e-4, 54.6] log-concentration units.
_LOG_SIGMA_BOUNDS = (-8.0, 4.0)
_SLOPE_BOUND = 50.0
_INTERCEPT_HALF_RANGE = 15.0

# Below this total weight the grid point has effectively no data.
_MIN_TOTAL_WEIGHT = 1e-6
_MIN_VALID_ROWS = 3


def _log_prob_between(z_low: np.ndarray, z_high: np.ndarray) -> np.ndarray:
    """log(Phi(z_high) - Phi(z_low)) for z_low < z_high, accurate in both tails.

    In the upper tail both Phi values are close to 1 and their difference
    loses precision, so there the identity
    ``Phi(b) - Phi(a) = Phi(-a) - Phi(-b)`` is used instead.
    """
    upper = z_low > 0
    big = np.where(upper, -z_low, z_high)
    small = np.where(upper, -z_high, z_low)
    log_big = log_ndtr(big)
    log_small = log_ndtr(small)
    with np.errstate(divide="ignore"):
        result = log_big + np.log1p(-np.exp(log_small - log_big))
    return result


def _neg_log_likelihood_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Weighted interval-censored Gaussian negative log-likelihood and gradient.

    This is the likelihood ``survreg(Surv(log(ConcLow), log(ConcHigh),
    type = "interval2") ~ ...)`` maximises in R EGRET. Each observation is
    an interval ``[y_low, y_high]`` of log-concentration:

    - ``y_low == y_high``: a detected value; contributes the density.
    - ``y_low == -inf``: left-censored (a non-detect, ConcLow = 0);
      contributes ``Phi(z_high)``.
    - ``y_high == +inf``: right-censored; contributes ``1 - Phi(z_low)``.
    - otherwise: interval-censored; contributes ``Phi(z_high) - Phi(z_low)``.

    ``params`` is ``[beta_0, ..., beta_k, log_sigma]`` and *X is assumed
    centered on the target point*. With ``mu = X @ beta`` and
    ``z = (y - mu) / sigma``, the per-observation derivatives are

    ============  ===============================  ==================================
    term          sigma * d ll / d mu              d ll / d log_sigma
    ============  ===============================  ==================================
    detected      ``z``                            ``z**2 - 1``
    censored      ``-(phi(zH) - phi(zL)) / P``     ``-(zH phi(zH) - zL phi(zL)) / P``
    ============  ===============================  ==================================

    where ``P = Phi(zH) - Phi(zL)`` and ``phi(+-inf) = 0``.

    Returns:
        ``(nll, gradient)``: the negative log-likelihood and its gradient
        with respect to ``params``.
    """
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    mu = X @ beta

    detected = y_low == y_high
    censored = ~detected

    ll = np.empty(len(mu))
    g = np.empty(len(mu))
    h = np.empty(len(mu))

    # -- detected values ---------------------------------------------------
    z = (y_high[detected] - mu[detected]) / sigma
    ll[detected] = -log_sigma - _HALF_LOG_2PI - 0.5 * z ** 2
    g[detected] = z
    h[detected] = z ** 2 - 1.0

    # -- censored values (left, right or interval) --------------------------
    z_high = (y_high[censored] - mu[censored]) / sigma
    z_low = (y_low[censored] - mu[censored]) / sigma
    log_p = _log_prob_between(z_low, z_high)
    ll[censored] = log_p

    # phi(z) / P, computed in log space so it stays finite far into the tails.
    with np.errstate(over="ignore", invalid="ignore"):
        ratio_high = np.exp(-_HALF_LOG_2PI - 0.5 * z_high ** 2 - log_p)
        ratio_low = np.exp(-_HALF_LOG_2PI - 0.5 * z_low ** 2 - log_p)
        z_ratio_high = np.where(np.isfinite(z_high), z_high * ratio_high, 0.0)
        z_ratio_low = np.where(np.isfinite(z_low), z_low * ratio_low, 0.0)
    g[censored] = -(ratio_high - ratio_low)
    h[censored] = -(z_ratio_high - z_ratio_low)

    wg = weights * g
    grad_beta = -(X.T @ wg) / sigma
    grad_log_sigma = -float(np.sum(weights * h))

    nll = -float(np.sum(weights * ll))
    if not np.isfinite(nll):
        # Hand the optimiser a large but finite value with a zero gradient
        # rather than a NaN it cannot recover from.
        return 1e12, np.zeros_like(params)

    grad = np.concatenate([grad_beta, [grad_log_sigma]])
    grad = np.where(np.isfinite(grad), grad, 0.0)
    return nll, grad


def _log_bounds(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Log-concentration interval for each sample, as ``survreg`` sees it.

    Uses ``ConcLow`` / ``ConcHigh`` when present: ``log(0) = -inf`` marks a
    left-censored value and a missing ``ConcHigh`` a right-censored one,
    exactly as ``Surv(..., type = "interval2")`` reads them. Samples without
    those columns fall back to ``LogConcHigh`` with ``Uncen``.
    """
    if "ConcLow" in sample.columns and "ConcHigh" in sample.columns:
        conc_low = sample["ConcLow"].to_numpy(dtype=float)
        conc_high = sample["ConcHigh"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y_low = np.where(conc_low > 0, np.log(conc_low), -np.inf)
            # A missing or zero upper bound is a right-censored value.
            has_high = np.isfinite(conc_high) & (conc_high > 0)
            y_high = np.where(has_high, np.log(conc_high), np.inf)
        # A row with neither bound carries no information.
        no_bounds = ~np.isfinite(y_low) & ~np.isfinite(y_high)
        y_high = np.where(no_bounds, np.nan, y_high)
        return y_low, y_high

    y_high = sample["LogConcHigh"].to_numpy(dtype=float)
    uncen = sample["Uncen"].to_numpy(dtype=float)
    y_low = np.where(uncen == 1, y_high, -np.inf)
    return y_low, y_high


def _neg_log_likelihood_only(params, X, y_low, y_high, weights) -> float:
    """The negative log-likelihood alone, for the derivative-free fallback."""
    nll, _ = _neg_log_likelihood_and_grad(params, X, y_low, y_high, weights)
    return nll


def _wls_init(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares starting beta.

    Solves ``min_beta ||diag(sqrt(w)) (X beta - y)||^2``.  Falls back to an
    intercept-only fit (weighted mean, zero slopes) if the system is
    rank-deficient, which happens when the weighted subset spans too little
    of the design space.
    """
    w_sqrt = np.sqrt(w)
    A = X * w_sqrt[:, None]
    b = y * w_sqrt
    try:
        beta, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < X.shape[1] or not np.all(np.isfinite(beta)):
            beta = np.zeros(X.shape[1])
            beta[0] = float(np.average(y, weights=w))
    except np.linalg.LinAlgError:
        beta = np.zeros(X.shape[1])
        beta[0] = float(np.average(y, weights=w))
    return beta


def run_surv_reg(
    target_year: float,
    target_logQ: float,
    target_sinDY: float,
    target_cosDY: float,
    sample: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Fit a weighted censored log-normal regression at one WRTDS grid point.

    Mirrors R ``runSurvReg()``.  Called once per grid point by
    :func:`~pyegret.model.est_surfaces.est_surfaces`.

    Args:
        target_year: Decimal year of the grid point.
        target_logQ: log(Q) of the grid point.
        target_sinDY: sin(2*pi * target_year).
        target_cosDY: cos(2*pi * target_year).
        sample: Sample DataFrame with columns ``DecYear``, ``LogQ``,
            ``SinDY``, ``CosDY``, and ``ConcLow`` / ``ConcHigh`` (or, failing
            those, ``LogConcHigh`` / ``Uncen``).
        weights: 1-D array of tricube weights, one per sample row. Use
            :func:`~pyegret.model.tri_cube.compute_adaptive_weights` to build
            these - a fixed-window kernel leaves sparse grid points with too
            few observations to support a fit.

    Returns:
        ``(yHat, SE, ConcHat)`` where:

        - ``yHat``    - fitted log-concentration at the target point
        - ``SE``      - fitted sigma (sd of log-concentration residuals)
        - ``ConcHat`` - ``exp(yHat + 0.5 * SE**2)``, the bias-corrected
          concentration

        Returns ``(nan, nan, nan)`` only when the weighted sample genuinely
        cannot support an estimate: no observation carries meaningful weight,
        or fewer than three usable rows remain.
    """
    weights = np.asarray(weights, dtype=float)

    if np.sum(weights) < _MIN_TOTAL_WEIGHT:
        return np.nan, np.nan, np.nan

    # -- Design matrix, centered on the target -----------------------------
    n = len(sample)
    X = np.column_stack([
        np.ones(n),
        sample["DecYear"].to_numpy(dtype=float) - target_year,
        sample["LogQ"].to_numpy(dtype=float) - target_logQ,
        sample["SinDY"].to_numpy(dtype=float) - target_sinDY,
        sample["CosDY"].to_numpy(dtype=float) - target_cosDY,
    ])
    y_low, y_high = _log_bounds(sample)

    # Keep only rows that carry weight and have usable values. Dropping
    # zero-weight rows shrinks the problem without changing the likelihood.
    usable = ~np.isnan(y_high) & ~np.isnan(y_low)
    valid = usable & np.isfinite(weights) & (weights > 0)
    if np.sum(valid) < _MIN_VALID_ROWS:
        return np.nan, np.nan, np.nan

    X = X[valid]
    y_low = y_low[valid]
    y_high = y_high[valid]
    w = weights[valid]

    # Starting values use one representative value per interval: the upper
    # bound, or the lower bound when there is no upper bound.
    y = np.where(np.isfinite(y_high), y_high, y_low)

    # -- Starting point ----------------------------------------------------
    y_mean = float(np.average(y, weights=w))
    beta0 = _wls_init(X, y, w)
    resid0 = y - X @ beta0
    sigma0 = float(np.sqrt(max(np.average(resid0 ** 2, weights=w), 1e-4)))
    x0 = np.concatenate([beta0, [np.log(sigma0)]])

    bounds = (
        [(y_mean - _INTERCEPT_HALF_RANGE, y_mean + _INTERCEPT_HALF_RANGE)]
        + [(-_SLOPE_BOUND, _SLOPE_BOUND)] * (X.shape[1] - 1)
        + [_LOG_SIGMA_BOUNDS]
    )
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    args = (X, y_low, y_high, w)
    best = None

    try:
        best = minimize(
            _neg_log_likelihood_and_grad, x0, args=args, jac=True,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
    except Exception:
        best = None

    # -- Fallback: derivative-free, for the rare non-convergence -----------
    if best is None or not np.all(np.isfinite(best.x)):
        try:
            best = minimize(
                _neg_log_likelihood_only,
                x0, args=args, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4,
                         "adaptive": True},
            )
        except Exception:
            return np.nan, np.nan, np.nan

    if best is None or not np.all(np.isfinite(best.x)):
        return np.nan, np.nan, np.nan

    # In the centered system the intercept is the fit at the target.
    yHat = float(best.x[0])
    sigma = float(np.exp(best.x[-1]))
    conc_hat = float(np.exp(yHat + 0.5 * sigma ** 2))

    if not (np.isfinite(yHat) and np.isfinite(sigma) and np.isfinite(conc_hat)):
        return np.nan, np.nan, np.nan

    return yHat, sigma, conc_hat

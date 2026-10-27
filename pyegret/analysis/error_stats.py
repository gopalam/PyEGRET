"""
Model performance diagnostics. Mirrors R ``errorStats()`` and ``fluxBiasStat()``.

``error_stats``     R-squared of log concentration and log flux, RMSE and the
                    standard error of prediction in percent, from the
                    cross-validated residuals with censored values randomised.
``flux_bias_stat``  the three flux-bias numbers EGRET reports (bias1, bias2,
                    bias3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.data.quality import make_augmented_sample

__all__ = ["error_stats", "flux_bias_stat"]


def error_stats(elist: dict, seed: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Goodness-of-fit statistics for a fitted WRTDS model.

    Mirrors R ``errorStats()``. Censored samples are first replaced by random
    draws from the fitted distribution, truncated to the reported interval
    (:func:`~pyegret.data.quality.make_augmented_sample`), so every sample has a
    residual. Then, with ``r`` the residual in natural-log units::

        RsqLogC    = (var(log C) - var(r)) / var(log C)
        RsqLogF    = (var(log F) - var(r)) / var(log F)     F = C * Q * 86.4
        rmse       = sqrt(var(r))
        sepPercent = 100 * sqrt(exp(var(r)) - 1)

    Variances use the n - 1 denominator, as R's ``var`` does. Because of the
    random draws the numbers move slightly from run to run when the record
    has censored values; pass ``seed`` to fix them.

    Args:
        elist: eList dict whose Sample has ``yHat`` and ``SE`` (run
            :func:`~pyegret.model.model_estimation.model_estimation` first).
        seed: Seed for the censored-value draws. None draws fresh.
        verbose: Print the four statistics, as R does. Default True.

    Returns:
        One-row DataFrame with columns ``RsqLogC``, ``RsqLogF``, ``rmse`` and
        ``sepPercent``.

    Raises:
        ValueError: If Sample has no ``yHat`` / ``SE`` columns.
    """
    sample = elist["Sample"]
    if sample is None or "yHat" not in sample.columns or "SE" not in sample.columns:
        raise ValueError(
            "Sample has no 'yHat' / 'SE' columns - run model_estimation() first."
        )

    augmented = make_augmented_sample(elist, seed=seed)
    sample = augmented["Sample"]

    r_resid = sample["rResid"].to_numpy(dtype=float)
    r_observed = sample["rObserved"].to_numpy(dtype=float)
    q = sample["Q"].to_numpy(dtype=float)

    log_conc = np.log(r_observed)
    log_flux = np.log(r_observed * q * 86.4)

    ok = np.isfinite(r_resid) & np.isfinite(log_conc) & np.isfinite(log_flux)
    r_resid = r_resid[ok]
    log_conc = log_conc[ok]
    log_flux = log_flux[ok]

    var_lc = np.var(log_conc, ddof=1)
    var_resid = np.var(r_resid, ddof=1)
    var_lf = np.var(log_flux, ddof=1)

    rsq_log_c = (var_lc - var_resid) / var_lc
    rsq_log_f = (var_lf - var_resid) / var_lf
    rmse = np.sqrt(var_resid)
    sep_percent = 100.0 * np.sqrt(np.exp(var_resid) - 1.0)

    if verbose:
        print(f"\n Root Mean Squared Error in natural log units =  {rmse:.3g}")
        print(f" Rsquared for natural log of concentration    =  {rsq_log_c:.3g}")
        print(f" Rsquared for natural log of flux             =  {rsq_log_f:.3g}")
        print(f" Standard error of estimate = {sep_percent:.3g} %")

    result = pd.DataFrame({
        "RsqLogC": [float(rsq_log_c)],
        "RsqLogF": [float(rsq_log_f)],
        "rmse": [float(rmse)],
        "sepPercent": [float(sep_percent)],
    })
    return result


def flux_bias_stat(sample: pd.DataFrame) -> pd.DataFrame:
    """The flux bias statistic. Mirrors R ``fluxBiasStat()``.

    Compares the model's estimate of flux on sampled days with the observed
    flux, using each end of the censoring interval in turn::

        bias1 = (sum(ConcHat*Q) - sum(ConcHigh*Q)) / sum(ConcHat*Q)
        bias2 = (sum(ConcHat*Q) - sum(ConcLow*Q))  / sum(ConcHat*Q)
        bias3 = (bias1 + bias2) / 2

    With no censoring, bias1 = bias2 = bias3. A value near zero means the
    model is not systematically over- or under-estimating flux; EGRET's
    guidance treats magnitudes above about 0.1 as a warning sign.

    Args:
        sample: Sample DataFrame from a fitted eList (``elist["Sample"]``),
            with ``ConcLow``, ``ConcHigh``, ``ConcHat`` and ``Q``. R takes
            the Sample, not the eList; an eList dict is accepted too.

    Returns:
        One-row DataFrame with columns ``bias1``, ``bias2`` and ``bias3``.

    Raises:
        ValueError: If Sample has no ``ConcHat`` column.
    """
    if isinstance(sample, dict):
        sample = sample["Sample"]
    if sample is None or "ConcHat" not in sample.columns:
        raise ValueError(
            "Sample has no 'ConcHat' column - run model_estimation() first."
        )

    q = sample["Q"].to_numpy(dtype=float)
    conc_low = sample["ConcLow"].to_numpy(dtype=float)
    conc_high = sample["ConcHigh"].to_numpy(dtype=float)
    conc_hat = sample["ConcHat"].to_numpy(dtype=float)

    sum_low = np.nansum(conc_low * q)
    sum_high = np.nansum(conc_high * q)
    sum_est = np.nansum(conc_hat * q)

    bias1 = (sum_est - sum_high) / sum_est
    bias2 = (sum_est - sum_low) / sum_est
    bias3 = (bias1 + bias2) / 2.0

    result = pd.DataFrame({
        "bias1": [float(bias1)],
        "bias2": [float(bias2)],
        "bias3": [float(bias3)],
    })
    return result

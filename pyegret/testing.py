"""
Synthetic records with a known answer, for testing the WRTDS engine.

Real streamgage data cannot tell you whether a trend estimate is *correct* -
only whether it looks plausible. This module builds records whose true
concentration surface is prescribed, so estimation error can be measured
directly.

The truth model is log-linear in the same four predictors WRTDS regresses on::

    log C = a + b*(year - 2000) + c*logQ + d*sin(2*pi*t) + e*cos(2*pi*t) + eps

with ``eps ~ N(0, sigma^2)``. Because the errors are log-normal, the expected
concentration at a point is ``exp(log C + sigma^2/2)`` - the same bias
correction WRTDS applies - so estimates should be compared against
:func:`truth_conc`, not ``exp(truth_log_conc(...))``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.data.populate import populate_date_columns, populate_sample_columns
from pyegret.elist import make_elist

__all__ = ["make_truth", "truth_change_over", "DEFAULT_TRUTH", "annual_mean_by_calendar_year", "truth_log_conc", "truth_conc",
           "make_synthetic_elist"]


def make_truth(
    intercept: float = -1.0,
    trend: float = -0.02,
    logq: float = 0.30,
    sin_amp: float = 0.40,
    cos_amp: float = -0.25,
    sigma: float = 0.15,
) -> dict:
    """Coefficients of the known concentration surface, as a dict.

    Args:
        intercept: Log-concentration at year 2000, logQ 0, at the seasonal origin.
        trend: Change in log-concentration per year. -0.02 is a 2 %/yr decline.
        logq: Slope on log-discharge.
        sin_amp: Coefficient on sin(2*pi*t).
        cos_amp: Coefficient on cos(2*pi*t).
        sigma: Standard deviation of the log-scale residuals.

    Returns:
        Dict with keys ``intercept``, ``trend``, ``logq``, ``sin_amp``,
        ``cos_amp`` and ``sigma``.
    """
    truth = {
        "intercept": intercept,
        "trend": trend,
        "logq": logq,
        "sin_amp": sin_amp,
        "cos_amp": cos_amp,
        "sigma": sigma,
    }
    return truth


def truth_change_over(truth: dict, years: float) -> float:
    """Fractional change in concentration over ``years`` years."""
    return float(np.exp(truth["trend"] * years) - 1.0)


DEFAULT_TRUTH = make_truth()


def truth_log_conc(dec_year, log_q, truth: dict = DEFAULT_TRUTH):
    """Mean log-concentration under the truth model.

    Args:
        dec_year: Decimal year, scalar or array.
        log_q: Natural log of discharge, scalar or array.
        truth: Coefficients. Defaults to :data:`DEFAULT_TRUTH`.

    Returns:
        Mean log-concentration, same shape as the inputs.
    """
    dec_year = np.asarray(dec_year, dtype=float)
    log_q = np.asarray(log_q, dtype=float)
    return (truth["intercept"]
            + truth["trend"] * (dec_year - 2000.0)
            + truth["logq"] * log_q
            + truth["sin_amp"] * np.sin(2 * np.pi * dec_year)
            + truth["cos_amp"] * np.cos(2 * np.pi * dec_year))


def truth_conc(dec_year, log_q, truth: dict = DEFAULT_TRUTH):
    """Expected concentration under the truth model.

    Includes the log-normal bias correction ``exp(sigma^2/2)``, so this is
    what a correct WRTDS ``ConcDay`` should reproduce.

    Args:
        dec_year: Decimal year, scalar or array.
        log_q: Natural log of discharge, scalar or array.
        truth: Coefficients. Defaults to :data:`DEFAULT_TRUTH`.

    Returns:
        Expected concentration, same shape as the inputs.
    """
    return np.exp(truth_log_conc(dec_year, log_q, truth) + 0.5 * truth["sigma"] ** 2)


def make_synthetic_elist(
    start: str = "1990-01-01",
    end: str = "2009-12-31",
    n_samples: int = 240,
    truth: dict = DEFAULT_TRUTH,
    censor_fraction: float = 0.0,
    step_change: float = 0.0,
    step_date: str | None = None,
    flow_trend: float = 0.0,
    seed: int = 0,
) -> dict:
    """Build an eList whose true concentration surface is known.

    Discharge is a seasonal cycle plus a random walk, which gives the
    within-year and between-year flow variation WRTDS is designed to separate.
    Samples are drawn uniformly at random from the daily record.

    Args:
        start: First day of the daily record, ``"YYYY-MM-DD"``.
        end: Last day of the daily record.
        n_samples: Number of water-quality samples to draw.
        truth: Coefficients of the true surface.
        censor_fraction: Fraction of samples to mark as left-censored at
            their observed value. 0 gives a fully detected record.
        step_change: Abrupt shift in log-concentration applied from
            ``step_date`` onward, on top of the gradual trend. -0.4 is a
            33 % drop, the sort of thing a treatment upgrade produces. Use
            with ``step_date`` to build a record the wall should recover.
        step_date: Date the step takes effect, ``"YYYY-MM-DD"``. Required
            when ``step_change`` is non-zero.
        flow_trend: Trend in log discharge per year, applied to the daily
            record. Non-zero makes the flow distribution non-stationary,
            which is what generalized flow normalization exists to handle.
            The concentration-discharge relationship is left untouched, so
            any resulting change in load is attributable to discharge alone.
        seed: Random seed, for reproducibility.

    Returns:
        An eList with populated Daily and Sample DataFrames and a minimal
        INFO row.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n)
    random_walk = np.cumsum(rng.standard_normal(n))
    log_q = (0.6 * np.sin(2 * np.pi * t / 365.25 - 1.0)
             + 0.8 * random_walk / np.sqrt(n) * 3)
    log_q -= log_q.mean()
    if flow_trend != 0.0:
        log_q = log_q + flow_trend * (t / 365.25)

    daily = pd.DataFrame({"Date": dates, "Q": np.exp(log_q), "LogQ": log_q})
    daily = populate_date_columns(daily)

    idx = np.sort(rng.choice(n, size=n_samples, replace=False))
    picked = daily.iloc[idx].reset_index(drop=True)
    log_conc = truth_log_conc(
        picked["DecYear"].to_numpy(), picked["LogQ"].to_numpy(), truth
    )
    if step_change != 0.0:
        if step_date is None:
            raise ValueError("step_change needs a step_date.")
        log_conc = log_conc + step_change * (
            picked["Date"].to_numpy() > np.datetime64(pd.Timestamp(step_date))
        )
    conc = np.exp(log_conc + truth["sigma"] * rng.standard_normal(len(picked)))

    uncen = np.ones(len(picked), dtype=int)
    if censor_fraction > 0:
        censored = rng.random(len(picked)) < censor_fraction
        uncen[censored] = 0

    sample = pd.DataFrame({
        "Date": picked["Date"],
        "ConcLow": np.where(uncen == 1, conc, 0.0),
        "ConcHigh": conc,
        "Uncen": uncen,
    })
    sample = populate_date_columns(sample)
    sample["Q"] = picked["Q"].to_numpy()
    sample["LogQ"] = picked["LogQ"].to_numpy()
    sample = populate_sample_columns(sample)

    info = pd.DataFrame([{
        "shortName": "Synthetic River",
        "paramShortName": "Synthetic Constituent",
        "param_cd": "00000",
        "station_nm": "Synthetic River",
        "site_no": "00000000",
        "drainSqKm": 100.0,
        "staAbbrev": "SYN",
        "constitAbbrev": "SYN",
    }])

    return make_elist(INFO=info, Daily=daily, Sample=sample)


def annual_mean_by_calendar_year(daily: pd.DataFrame, column: str) -> pd.Series:
    """Mean of a Daily column for each calendar year, indexed by year.

    A quick summary for tests and diagnostics; for EGRET's own annual
    results use :func:`pyegret.analysis.setup_years`.
    """
    years = np.floor(daily["DecYear"].to_numpy(dtype=float))
    years = np.asarray(years, dtype=int)
    values = daily[column].to_numpy(dtype=float)
    series = pd.Series(values)
    groups = series.groupby(years)
    return groups.mean()

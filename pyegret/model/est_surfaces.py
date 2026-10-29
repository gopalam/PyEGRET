"""
WRTDS surface estimation: a grid of censored regressions over (year, logQ).

Grid geometry follows R EGRET's ``estSurfaces()``:

- **Year axis** runs from ``floor(min(Daily.DecYear))`` to
  ``ceil(max(Daily.DecYear))`` in steps of 1/16 year (about 23 days), as
  R's ``surfaceIndex()`` builds it.
- **log-Q axis** holds 14 points spanning the *Daily* discharge range,
  padded by 0.05 log units at each end.

Both details matter.

The year step is not cosmetic.  Season enters the regression through
``SinDY`` and ``CosDY`` evaluated at the grid year, so a grid coarser than
the seasonal cycle fits the seasonal terms and then interpolates them
straight back out.  Sixteen points per year resolves it; sixteen points
across the whole record does not.

The log-Q axis comes from Daily rather than Sample because the surface is
later interpolated onto every day in the record.  Sampling rarely covers the
full range of discharge, so a Sample-derived axis would force extrapolation
on exactly the high- and low-flow days that dominate flux.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.elist import has_sample
from pyegret.model.run_surv_reg import run_surv_reg
from pyegret.model.tri_cube import (
    DEFAULT_MIN_NUM_OBS,
    DEFAULT_MIN_NUM_UNCEN,
    DEFAULT_WINDOW_Q,
    DEFAULT_WINDOW_S,
    DEFAULT_WINDOW_Y,
    compute_adaptive_weights,
)

__all__ = ["est_surfaces", "est_surfaces_on_grid", "build_grid_axes", "decimal_high_low",
           "run_surv_reg_points"]

_TWO_PI = 2.0 * np.pi

# EGRET: stepYear = 1/16, nVectorLogQ = 14, logQ padding = 0.05
POINTS_PER_YEAR = 16
N_VECTOR_LOGQ = 14
_LOGQ_PAD = 0.05


def build_grid_axes(
    elist: dict,
    points_per_year: int = POINTS_PER_YEAR,
    n_vector_logQ: int = N_VECTOR_LOGQ,
    n_vector_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (year, logQ) axes the surface is estimated on.

    Args:
        elist: eList dict with Daily and Sample DataFrames.
        points_per_year: Year-axis resolution. Default 16, matching EGRET.
        n_vector_logQ: Number of log-discharge grid points. Default 14.
        n_vector_year: Override the year axis with exactly this many points
            spanning the same range. Intended for fast tests; leave None for
            the EGRET-resolution grid.

    Returns:
        ``(vector_year, vector_logQ)``.
    """
    daily = elist["Daily"]

    bottom_year = float(np.floor(np.nanmin(daily["DecYear"].to_numpy(dtype=float))))
    top_year = float(np.ceil(np.nanmax(daily["DecYear"].to_numpy(dtype=float))))

    if n_vector_year is not None:
        vector_year = np.linspace(bottom_year, top_year, int(n_vector_year))
    else:
        n = int(round((top_year - bottom_year) * points_per_year)) + 1
        vector_year = bottom_year + np.arange(n) / float(points_per_year)

    log_q = daily["LogQ"].to_numpy(dtype=float)
    log_q = log_q[np.isfinite(log_q)]
    bottom_logQ = float(np.min(log_q)) - _LOGQ_PAD
    top_logQ = float(np.max(log_q)) + _LOGQ_PAD
    vector_logQ = np.linspace(bottom_logQ, top_logQ, int(n_vector_logQ))

    return vector_year, vector_logQ


def decimal_high_low(sample: pd.DataFrame) -> tuple[float, float]:
    """Water-year bounds of a sample record, for edge adjustment.

    Mirrors R ``decimalHighLow()``. Returns the start of the water year
    containing the first sample and the end of the one containing the last,
    rather than the raw sample extremes.

    Args:
        sample: Sample DataFrame with ``DecYear`` and ``Month``, in date order.

    Returns:
        ``(dec_low, dec_high)``.
    """
    dec_year = sample["DecYear"].to_numpy(dtype=float)
    month = sample["Month"].to_numpy(dtype=int)

    dec_low = np.trunc(dec_year[0]) - 0.25
    dec_high = np.trunc(dec_year[-1]) + 0.75
    if month[0] in (10, 11, 12):
        dec_low += 1
    if month[-1] in (10, 11, 12):
        dec_high += 1
    return float(dec_low), float(dec_high)


def est_surfaces_on_grid(
    sample: pd.DataFrame,
    vector_year: np.ndarray,
    vector_logQ: np.ndarray,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    edge_adjust: bool = True,
    verbose: bool = True,
    dec_low: float | None = None,
    dec_high: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the WRTDS regression at every point of a given (year, logQ) grid.

    The engine behind :func:`est_surfaces`, with the grid passed in so a
    caller can fit part of a surface (R ``estSurfaces(surfaceStart = ,
    surfaceEnd = , localSample = )``, as ``runPairs`` and ``runGroups`` do).

    Args:
        sample: Sample DataFrame to fit from, in date order.
        vector_year: Decimal-year axis.
        vector_logQ: Log-discharge axis.
        window_y, window_q, window_s: Starting half-windows.
        min_num_obs, min_num_uncen: Minimum weighted observations.
        edge_adjust: Widen the time window near the ends of the sample
            record, whose ends are taken from :func:`decimal_high_low`.
        verbose: Print progress.
        dec_low, dec_high: The record ends used for edge adjustment, when
            they should not come from the sample (EGRETci's ``wBT`` uses the
            first and last day of Daily, as R ``runSurvReg`` allows).

    Returns:
        ``(surfaces, windows_used)``: the ``(n_year, n_logQ, 3)`` surface
        and the time half-window each grid point ended up using.
    """
    # Mirrors R runSurvReg(): samples without discharge are set aside, and the
    # fit is refused when the minimums cannot be met. R's test is ">=".
    has_q = ~np.isnan(sample["LogQ"].to_numpy(dtype=float))
    if not np.all(has_q):
        sample = sample[has_q]
        sample = sample.reset_index(drop=True)
    if min_num_uncen >= np.sum(sample["Uncen"].to_numpy()):
        raise ValueError("min_num_uncen is greater than total number of samples")
    if min_num_obs >= len(sample):
        raise ValueError("min_num_obs is greater than total number of samples")

    s_years = sample["DecYear"].to_numpy(dtype=float)
    s_logQ = sample["LogQ"].to_numpy(dtype=float)
    s_uncen = sample["Uncen"].to_numpy()
    sample_low, sample_high = decimal_high_low(sample)
    if dec_low is None:
        dec_low = sample_low
    if dec_high is None:
        dec_high = sample_high

    n_yr = len(vector_year)
    n_lq = len(vector_logQ)
    surfaces = np.full((n_yr, n_lq, 3), np.nan)
    windows_used = np.full((n_yr, n_lq), np.nan)

    total = n_yr * n_lq
    done = 0
    next_report = 10
    if verbose:
        print(f"Estimating surfaces on a {n_yr} x {n_lq} grid "
              f"({total} points)...", flush=True)

    for i in range(n_yr):
        yr = float(vector_year[i])
        sin_dy = np.sin(_TWO_PI * yr)
        cos_dy = np.cos(_TWO_PI * yr)
        for j in range(n_lq):
            lq = float(vector_logQ[j])
            weights, used = compute_adaptive_weights(
                target_year=yr,
                target_logQ=lq,
                sample_years=s_years,
                sample_logQ=s_logQ,
                sample_uncen=s_uncen,
                window_y=window_y,
                window_q=window_q,
                window_s=window_s,
                min_num_obs=min_num_obs,
                min_num_uncen=min_num_uncen,
                edge_adjust=edge_adjust,
                dec_low=dec_low,
                dec_high=dec_high,
            )
            windows_used[i, j] = used[0]
            surfaces[i, j, :] = run_surv_reg(yr, lq, sin_dy, cos_dy, sample, weights)

            done = done + 1
            if verbose:
                pct = int(100 * done / total)
                if pct >= next_report:
                    print(f"  {next_report}% complete", flush=True)
                    next_report = next_report + 10

    n_nan = int(np.sum(np.isnan(surfaces[:, :, 0])))
    if verbose and n_nan:
        print(f"  {n_nan} of {total} grid points could not be estimated.", flush=True)
    return surfaces, windows_used


def est_surfaces(
    elist: dict,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    edge_adjust: bool = True,
    points_per_year: int = POINTS_PER_YEAR,
    n_vector_logQ: int = N_VECTOR_LOGQ,
    n_vector_year: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Estimate the WRTDS concentration surface over a (year, logQ) grid.

    Mirrors R ``estSurfaces()``.  At each grid point a tricube-weighted
    censored log-normal regression is fit by
    :func:`~pyegret.model.run_surv_reg.run_surv_reg`, using windows that expand
    until enough observations are in range
    (:func:`~pyegret.model.tri_cube.compute_adaptive_weights`).

    Args:
        elist: eList dict with valid Daily and Sample DataFrames.
        window_y: Starting tricube half-window for time, in years. Default 7.0.
        window_q: Starting half-window for log-discharge. Default 2.0.
        window_s: Starting half-window for season, in years. Default 0.5.
        min_num_obs: Minimum weighted observations per grid point. Default 100.
        min_num_uncen: Minimum uncensored weighted observations. Default 50.
        edge_adjust: Widen the time window near the record ends. Default True.
        points_per_year: Year-axis resolution. Default 16.
        n_vector_logQ: Number of log-discharge grid points. Default 14.
        n_vector_year: Override the year-axis point count. For fast tests.
        verbose: Print progress. Default True.

    Returns:
        ``ndarray`` of shape ``(n_year, n_logQ, 3)``:

        - ``[:, :, 0]`` - yHat, fitted log-concentration
        - ``[:, :, 1]`` - SE, sigma of log-concentration residuals
        - ``[:, :, 2]`` - ConcHat, ``exp(yHat + SE**2/2)``

        NaN marks a grid point the data could not support even after the
        windows expanded to cover the record.

    Side effects:
        Writes the grid axes into the eList passed in, as
        ``elist["vector_year"]`` and ``elist["vector_logQ"]``, and
        ``elist["window_y_used"]`` - the time half-window each grid point
        ended up needing. (R attaches the axes to the surfaces array as
        attributes; a NumPy array cannot carry them.)

    Raises:
        ValueError: If ``elist`` has no Sample data.
    """
    if not has_sample(elist):
        raise ValueError("eList has no Sample data; cannot estimate surfaces.")

    vector_year, vector_logQ = build_grid_axes(
        elist, points_per_year, n_vector_logQ, n_vector_year
    )
    surfaces, windows_used = est_surfaces_on_grid(
        elist["Sample"], vector_year, vector_logQ,
        window_y=window_y, window_q=window_q, window_s=window_s,
        min_num_obs=min_num_obs, min_num_uncen=min_num_uncen,
        edge_adjust=edge_adjust, verbose=verbose,
    )
    elist["vector_year"] = vector_year
    elist["vector_logQ"] = vector_logQ
    elist["window_y_used"] = windows_used
    return surfaces


def run_surv_reg_points(
    est_pt_year,
    est_pt_logq,
    dec_low: float,
    dec_high: float,
    sample: pd.DataFrame,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    edge_adjust: bool = True,
) -> np.ndarray:
    """Fit the WRTDS regression at any list of (year, logQ) points.

    Mirrors R ``runSurvReg(estPtYear, estPtLQ, DecLow, DecHigh, Sample, ...)``,
    which takes vectors of points; :func:`~pyegret.model.run_surv_reg.run_surv_reg`
    is the single-point version.

    Args:
        est_pt_year: Decimal years of the points.
        est_pt_logq: Natural-log discharges of the points (same length).
        dec_low, dec_high: Ends of the record, for edge adjustment.
        sample: Sample DataFrame to fit from.
        window_y, window_q, window_s, min_num_obs, min_num_uncen, edge_adjust:
            As for :func:`est_surfaces`.

    Returns:
        Array of shape ``(n_points, 3)``: yHat, SE, ConcHat.
    """
    years = np.asarray(est_pt_year, dtype=float)
    log_qs = np.asarray(est_pt_logq, dtype=float)
    s_years = sample["DecYear"].to_numpy(dtype=float)
    s_logQ = sample["LogQ"].to_numpy(dtype=float)
    s_uncen = sample["Uncen"].to_numpy()
    result = np.full((len(years), 3), np.nan)
    for k in range(len(years)):
        weights, _ = compute_adaptive_weights(
            target_year=years[k], target_logQ=log_qs[k],
            sample_years=s_years, sample_logQ=s_logQ, sample_uncen=s_uncen,
            window_y=window_y, window_q=window_q, window_s=window_s,
            min_num_obs=min_num_obs, min_num_uncen=min_num_uncen,
            edge_adjust=edge_adjust, dec_low=dec_low, dec_high=dec_high,
        )
        sin_dy = np.sin(_TWO_PI * years[k])
        cos_dy = np.cos(_TWO_PI * years[k])
        result[k, :] = run_surv_reg(years[k], log_qs[k], sin_dy, cos_dy, sample, weights)
    return result

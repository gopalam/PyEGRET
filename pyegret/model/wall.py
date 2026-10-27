"""
The wall: fitting a WRTDS surface across a known discontinuity.

WRTDS assumes change is gradual. Its smoother borrows strength from
neighbouring years on the argument that watersheds change through the
accumulated decisions of many actors, none of them abrupt. That assumption is
usually right and occasionally very wrong: a treatment plant upgrade, a
combined-sewer tunnel coming online, a dam built or removed, a stormwater
retrofit reaching completion. When concentration behaviour changes on a known
date, smoothing across that date averages two different rivers together and
hides the very effect you are trying to measure.

The wall splits the sample record at that date and fits the surface on each
side using only that side's data, so nothing before the wall informs the
estimate after it. Where the two fitted slices overlap they are averaged, so
the surface stays continuous even though the data behind it does not.

This is also a hypothesis test. Fit with and without the wall, compare, and
the difference is what the assumption of gradual change was costing you.

Mirrors R EGRET's ``stitch()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.elist import has_sample
from pyegret.model.est_surfaces import (
    decimal_high_low,
    N_VECTOR_LOGQ,
    POINTS_PER_YEAR,
    _LOGQ_PAD,
)
from pyegret.model.run_surv_reg import run_surv_reg
from pyegret.model.tri_cube import (
    DEFAULT_MIN_NUM_OBS,
    DEFAULT_MIN_NUM_UNCEN,
    DEFAULT_WINDOW_Q,
    DEFAULT_WINDOW_S,
    DEFAULT_WINDOW_Y,
    compute_adaptive_weights,
)

__all__ = ["stitch_surfaces", "decimal_high_low"]

_TWO_PI = 2.0 * np.pi

# EGRET relaxes the observation minimums when the record is cut in two,
# since each half necessarily holds fewer samples than the whole.
DEFAULT_FRACT_MIN = 0.75


def _decimal_date(date) -> float:
    """Decimal year of a date, matching EGRET's ``decimalDate``."""
    date = pd.Timestamp(date)
    year_start = pd.Timestamp(year=date.year, month=1, day=1)
    year_end = pd.Timestamp(year=date.year + 1, month=1, day=1)
    return date.year + (date - year_start) / (year_end - year_start)


def _fit_slice(
    sample: pd.DataFrame,
    vector_year: np.ndarray,
    vector_logQ: np.ndarray,
    indices: np.ndarray,
    windows: dict,
    minimums: dict,
    edge_adjust: bool,
) -> np.ndarray:
    """Fit the surface at the given year indices from one side's samples."""
    years = sample["DecYear"].to_numpy(dtype=float)
    log_q = sample["LogQ"].to_numpy(dtype=float)
    uncen = sample["Uncen"].to_numpy()
    dec_low, dec_high = decimal_high_low(sample)

    surface = np.full((len(vector_year), len(vector_logQ), 3), np.nan)
    for i in indices:
        year = float(vector_year[i])
        sin_dy, cos_dy = np.sin(_TWO_PI * year), np.cos(_TWO_PI * year)
        for j, lq in enumerate(vector_logQ):
            weights, _ = compute_adaptive_weights(
                target_year=year, target_logQ=float(lq),
                sample_years=years, sample_logQ=log_q, sample_uncen=uncen,
                edge_adjust=edge_adjust, dec_low=dec_low, dec_high=dec_high,
                **windows, **minimums,
            )
            surface[i, j, :] = run_surv_reg(
                year, float(lq), sin_dy, cos_dy, sample, weights
            )
    return surface


def stitch_surfaces(
    elist: dict,
    wall_date,
    surface_start=None,
    surface_end=None,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    fract_min: float = DEFAULT_FRACT_MIN,
    edge_adjust: bool = True,
    points_per_year: int = POINTS_PER_YEAR,
    n_vector_logQ: int = N_VECTOR_LOGQ,
    verbose: bool = True,
) -> np.ndarray:
    """Fit the WRTDS surface separately either side of a date, then join them.

    Mirrors R ``stitch()``.

    Args:
        elist: eList dict with Daily and Sample DataFrames.
        wall_date: The last day belonging to the period *before* the change,
            as a date or ``"YYYY-MM-DD"``. Samples from the next day onward
            form the second period. Choose it from what you know happened,
            not from the data.
        surface_start: First date the surface should cover. Defaults to the
            first sample.
        surface_end: Last date covered. Defaults to the last sample.
        window_y: Tricube half-window for time, in years. Default 7.0.
        window_q: Half-window for log-discharge. Default 2.0.
        window_s: Half-window for season, in years. Default 0.5.
        min_num_obs: Target minimum weighted observations per grid point.
            Relaxed automatically; see ``fract_min``.
        min_num_uncen: Target minimum uncensored observations. Relaxed to at
            most half of the effective ``min_num_obs``.
        fract_min: The observation minimum is reduced to this fraction of the
            smaller half's sample count when that is less than
            ``min_num_obs``. Default 0.75. Splitting a record leaves each
            side with fewer samples, and the original minimum could otherwise
            never be met.
        edge_adjust: Widen the time window near each period's ends.
            Default True.
        points_per_year: Year-axis resolution. Default 16.
        n_vector_logQ: Number of log-discharge grid points. Default 14.
        verbose: Report the split and the relaxed minimums. Default True.

    Returns:
        Surface array of shape ``(n_year, n_logQ, 3)``, the same layout
        :func:`~pyegret.model.est_surfaces.est_surfaces` returns, so it can be
        passed straight to
        :func:`~pyegret.model.est_daily.est_daily_from_surfaces`.

    Side effects:
        Sets ``elist["vector_year"]``, ``elist["vector_logQ"]`` and
        ``elist["wall_date"]``.

    Raises:
        ValueError: If the eList has no Sample, or either side of the wall is
            empty.
    """
    if not has_sample(elist):
        raise ValueError("eList has no Sample data; cannot fit a wall.")

    daily = elist["Daily"]
    sample = elist["Sample"].sort_values("Date")
    sample = sample.reset_index(drop=True)
    wall = pd.Timestamp(wall_date)

    # Grid axes span the Daily record, as EGRET's surfaceIndex does: the
    # surface is later read at every daily discharge, including the extremes.
    bottom_logQ = float(daily["LogQ"].min()) - _LOGQ_PAD
    top_logQ = float(daily["LogQ"].max()) + _LOGQ_PAD
    vector_logQ = np.linspace(bottom_logQ, top_logQ, n_vector_logQ)

    bottom_year = float(np.floor(daily["DecYear"].min()))
    top_year = float(np.ceil(daily["DecYear"].max()))
    n_year = int(round((top_year - bottom_year) * points_per_year)) + 1
    vector_year = bottom_year + np.arange(n_year) / float(points_per_year)

    dates = pd.to_datetime(sample["Date"])
    before = sample[dates <= wall]
    after = sample[dates > wall]
    if before.empty or after.empty:
        raise ValueError(
            f"The wall at {wall.date()} leaves {len(before)} samples before "
            f"and {len(after)} after; both sides need data."
        )

    # Relax the minimums to what the smaller half can actually supply.
    effective_obs = int(np.ceil(min(
        min_num_obs, fract_min * len(before), fract_min * len(after)
    )))
    effective_uncen = int(np.ceil(min(0.5 * effective_obs, min_num_uncen)))

    if verbose:
        print(f"Wall at {wall.date()}: {len(before)} samples before "
              f"({int((before['Uncen'] == 1).sum())} uncensored), "
              f"{len(after)} after "
              f"({int((after['Uncen'] == 1).sum())} uncensored)")
        print(f"  min_num_obs relaxed to {effective_obs}, "
              f"min_num_uncen to {effective_uncen}")

    start = pd.Timestamp(surface_start) if surface_start is not None else dates.iloc[0]
    end = pd.Timestamp(surface_end) if surface_end is not None else dates.iloc[-1]

    def _slice(low: float, high: float) -> np.ndarray:
        inside = np.flatnonzero((vector_year >= low) & (vector_year <= high))
        if inside.size == 0:
            return inside
        # Extend one grid point each way so the halves meet and overlap.
        first, last = max(inside[0] - 1, 0), min(inside[-1] + 1, len(vector_year) - 1)
        return np.arange(first, last + 1)

    indices_before = _slice(_decimal_date(start),
                            _decimal_date(wall + pd.Timedelta(days=1)))
    indices_after = _slice(_decimal_date(wall), _decimal_date(end))

    windows = {"window_y": window_y, "window_q": window_q, "window_s": window_s}
    minimums = {"min_num_obs": effective_obs, "min_num_uncen": effective_uncen}

    if verbose:
        print(f"  fitting {len(indices_before)} x {n_vector_logQ} grid points "
              f"before and {len(indices_after)} x {n_vector_logQ} after")

    surface_before = _fit_slice(before, vector_year, vector_logQ,
                                indices_before, windows, minimums, edge_adjust)
    surface_after = _fit_slice(after, vector_year, vector_logQ,
                               indices_after, windows, minimums, edge_adjust)

    combined = np.full((len(vector_year), n_vector_logQ, 3), np.nan)
    only_before = np.setdiff1d(indices_before, indices_after)
    only_after = np.setdiff1d(indices_after, indices_before)
    overlap = np.intersect1d(indices_before, indices_after)

    combined[only_before] = surface_before[only_before]
    combined[only_after] = surface_after[only_after]

    if overlap.size:
        # Average the two fits where they meet, then rebuild the
        # bias-corrected concentration from the averaged parts rather than
        # averaging it directly - exp() does not commute with a mean.
        y_hat = (surface_before[overlap, :, 0] + surface_after[overlap, :, 0]) / 2.0
        se = (surface_before[overlap, :, 1] + surface_after[overlap, :, 1]) / 2.0
        combined[overlap, :, 0] = y_hat
        combined[overlap, :, 1] = se
        combined[overlap, :, 2] = np.exp(y_hat + 0.5 * se ** 2)
        if verbose:
            print(f"  blended {overlap.size} overlapping year steps")

    elist["vector_year"] = vector_year
    elist["vector_logQ"] = vector_logQ
    elist["wall_date"] = wall

    n_nan = int(np.sum(np.isnan(combined[:, :, 0])))
    if verbose and n_nan:
        print(f"  {n_nan} of {combined[:, :, 0].size} grid points unestimated")

    return combined

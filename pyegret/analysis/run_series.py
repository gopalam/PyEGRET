"""
Annual flow-normalized series under generalized flow normalization.

Mirrors R ``runSeries()``: fit the surface over the span of the samples (with
a wall if asked), then flow-normalize each year against the discharges of its
own window (``window_side``) and/or either side of a flow break. The result
is an eList whose Daily carries ConcDay, FluxDay, FNConc and FNFlux, ready
for ``table_results``, ``table_change`` and the history plots.
"""
from __future__ import annotations

import numpy as np

from pyegret.analysis.run_groups import span_surface_elist
from pyegret.model.flex_fn import flex_fn

__all__ = ["run_series"]


def run_series(
    elist: dict,
    window_side: float,
    surface_start=None,
    surface_end=None,
    flow_break: bool = False,
    q1_end_date=None,
    q_start_date=None,
    q_end_date=None,
    wall: bool = False,
    old_surface: bool = False,
    sample1_end_date=None,
    sample_start_date=None,
    sample_end_date=None,
    pa_start: int | None = None,
    pa_long: int | None = None,
    fract_min: float = 0.75,
    min_num_obs: int = 100,
    min_num_uncen: int = 50,
    window_y: float = 7,
    window_q: float = 2,
    window_s: float = 0.5,
    edge_adjust: bool = True,
    verbose: bool = True,
) -> dict:
    """Flow-normalized daily and annual series with a moving flow window.

    Mirrors R ``runSeries()``.

    Args:
        elist: eList dict with Daily and Sample.
        window_side: Half-width in years of each year's flow-normalization
            window. 0 is stationary flow normalization. Required, as in R.
        surface_start, surface_end: Span to estimate. Default: whole periods
            of analysis covering the samples.
        flow_break, q1_end_date: An abrupt change in the discharge record and
            the last day before it.
        q_start_date, q_end_date: Limit the discharge record used.
        wall, sample1_end_date: Fit separate surfaces either side of
            ``sample1_end_date``.
        old_surface: Use the eList's existing surface.
        sample_start_date, sample_end_date: Limit the samples used.
        pa_start, pa_long: Period of analysis. Default: INFO, else 10/12.
        fract_min, min_num_obs, min_num_uncen: Minimum observations.
        window_y, window_q, window_s: Starting half-windows.
        edge_adjust: Widen the time window near the record ends. R's
            runSeries always passes TRUE to the surface fit; this is kept for
            symmetry with run_groups and defaults to True.
        verbose: Print progress.

    Returns:
        A new eList: Daily (trimmed to the discharge dates) with ``ConcDay``,
        ``FluxDay``, ``FNConc`` and ``FNFlux`` over the surface span (NaN
        elsewhere); the fitted surface and its axes; and INFO recording every
        setting, as R's ``runSeries`` writes them (``windowSide``,
        ``surfaceStart``, ``paStart`` and so on). The key ``"dateInfo"`` holds
        the flow-normalization segments, and ``"runSeries"`` is True (R's
        attribute of the same name).
    """
    min_num_uncen = int(np.ceil(min(0.5 * min_num_obs, min_num_uncen)))
    setup = span_surface_elist(
        elist, window_side, surface_start, surface_end, flow_break, q1_end_date,
        q_start_date, q_end_date, wall, old_surface, sample1_end_date,
        sample_start_date, sample_end_date, pa_start, pa_long, fract_min,
        min_num_obs, min_num_uncen, window_y, window_q, window_s, edge_adjust,
        verbose,
    )
    local = setup["elist"]
    daily = flex_fn(local, date_info=setup["dateInfo"])

    out = dict(local)
    out["Daily"] = daily
    out["dateInfo"] = setup["dateInfo"]
    info = out["INFO"].copy()
    dates = setup["dates"]
    info["wall"] = wall
    if not setup["oldSurface"]:
        info["surfaceStart"] = setup["surfaceStart"]
        info["surfaceEnd"] = setup["surfaceEnd"]
    info["sampleStartDate"] = dates["sampleStartDate"]
    info["sampleEndDate"] = dates["sampleEndDate"]
    info["sample1EndDate"] = sample1_end_date
    info["flowBreak"] = flow_break
    info["windowSide"] = window_side
    info["Q1EndDate"] = q1_end_date
    info["QStartDate"] = dates["QStartDate"]
    info["QEndDate"] = dates["QEndDate"]
    info["paLong"] = setup["paLong"]
    info["paStart"] = setup["paStart"]
    info["minNumUncen"] = min_num_uncen
    info["fractMin"] = fract_min
    info["minNumObs"] = min_num_obs
    info["windowQ"] = window_q
    info["windowY"] = window_y
    info["windowS"] = window_s
    info["edgeAdjust"] = edge_adjust
    out["INFO"] = info
    # R marks its result with attr(eList, "runSeries"); EGRETci checks for it.
    out["runSeries"] = True
    return out

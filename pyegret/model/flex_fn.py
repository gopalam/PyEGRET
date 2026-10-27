"""
Generalized flow normalization.

Standard flow normalization asks: what would concentration have been each
year if the discharge distribution were always the same? It answers by
averaging the fitted surface over the flow distribution of the *entire*
record, which removes year-to-year weather noise and leaves the watershed
signal.

That works only while the flow distribution itself is stationary. Over two or
three decades it often is not - a basin gets wetter, a dam changes the flow
duration curve, irrigation withdrawals grow - and then standard flow
normalization removes the real trend in discharge along with the noise, and
attributes the whole change to the watershed.

Generalized flow normalization (GFN) lets the reference flow distribution
move. Each stretch of the record is normalized against flows from its own
neighbourhood rather than the whole record. The trend then splits cleanly in
two, which :func:`~pyegret.analysis.run_pairs.run_pairs` reports:

- **CQTC**, the concentration-versus-discharge trend component: how the
  relationship between flow and concentration changed.
- **QTC**, the discharge trend component: how much of the change came from
  the flow distribution moving.

For a nutrient export study that distinction is the finding, not a detail. A
rise in load because the watershed leaks more nitrogen is a different result
from a rise because it rained more.

``window_side`` sets how far either side of each year the reference flows are
drawn from. Zero means the whole record - standard flow normalization, the
special case. Small values track non-stationarity closely but make the
normalized series jagged, since the curve moves as years enter and leave the
window; eleven is a common choice. ``flow_break_date`` handles the other
shape of non-stationarity: a single engineered step, like a dam completed on
a known date, where flows are stationary within each era but different
between them.

Mirrors R EGRET's ``flexFN`` and the ``dateInfo`` construction in
``runSeries``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.model.est_daily import est_daily_from_surfaces

__all__ = ["make_date_info", "flex_fn"]


def _as_date(value) -> pd.Timestamp:
    """Convert to a Timestamp at midnight."""
    stamp = pd.Timestamp(value)
    return stamp.normalize()


def _sliding_segments(
    surface_start: pd.Timestamp,
    surface_end: pd.Timestamp,
    q_start: pd.Timestamp,
    q_end: pd.Timestamp,
    window_side: int,
) -> pd.DataFrame:
    """One-year segments, each normalized by a window of surrounding flows.

    Mirrors R ``makeDateInfo()``. Near the ends of the discharge record the
    window cannot be centred, so it is shifted inward to keep its full width
    rather than truncated - otherwise the first and last years would be
    normalized against noticeably less data than the middle.
    """
    span_years = ((surface_end - surface_start).days + 1) / 365.0
    n_segments = int(np.ceil(span_years))

    flow_start = pd.DatetimeIndex(
        [surface_start + pd.DateOffset(years=i) for i in range(n_segments)]
    )
    if n_segments > 1:
        flow_end = pd.DatetimeIndex(
            [flow_start[1] + pd.DateOffset(years=i) - pd.Timedelta(days=1)
             for i in range(n_segments)]
        )
        flow_end = pd.DatetimeIndex(np.minimum(flow_end.values, surface_end.to_datetime64()))
    else:
        flow_end = pd.DatetimeIndex([surface_end])

    norm_start = pd.DatetimeIndex(
        [d - pd.DateOffset(years=window_side) for d in flow_start]
    )
    norm_end = pd.DatetimeIndex(
        [d + pd.DateOffset(years=window_side) for d in flow_end]
    )

    window_full = 1 + 2 * window_side
    norm_start_values = list(norm_start)
    norm_end_values = list(norm_end)
    for i in range(n_segments):
        if norm_start_values[i] < q_start:
            norm_end_values[i] = (
                q_start + pd.DateOffset(years=window_full) - pd.Timedelta(days=1)
            )
        if norm_end_values[i] > q_end:
            norm_start_values[i] = (
                q_end - pd.DateOffset(years=window_full) + pd.Timedelta(days=1)
            )
    clipped_start = []
    clipped_end = []
    for i in range(n_segments):
        clipped_start.append(max(norm_start_values[i], q_start))
        clipped_end.append(min(norm_end_values[i], q_end))
    norm_start = pd.DatetimeIndex(clipped_start)
    norm_end = pd.DatetimeIndex(clipped_end)

    frame = pd.DataFrame({
        "flowNormStart": norm_start,
        "flowNormEnd": norm_end,
        "flowStart": flow_start,
        "flowEnd": flow_end,
    })
    return frame[frame["flowStart"] < frame["flowEnd"]].reset_index(drop=True)


def make_date_info(
    elist: dict,
    window_side: int = 0,
    flow_break_date=None,
    surface_start=None,
    surface_end=None,
) -> pd.DataFrame:
    """Build the segment table that says how each stretch is normalized.

    Mirrors the ``dateInfo`` construction in R ``runSeries()``.

    Args:
        elist: eList dict with a Daily DataFrame.
        window_side: Half-width, in years, of the flow-normalization window
            either side of each year. 0 uses the whole record, which is
            standard flow normalization. 11 is a common non-zero choice.
        flow_break_date: Last day before an abrupt change in the discharge
            distribution - a dam completed, a diversion opened. Flows before
            and after are then never mixed. None means no break.
        surface_start: First date to produce results for. Defaults to the
            start of the Daily record.
        surface_end: Last date. Defaults to the end of the Daily record.

    Returns:
        DataFrame with ``flowNormStart``, ``flowNormEnd``, ``flowStart`` and
        ``flowEnd``. Each row says: for days between ``flowStart`` and
        ``flowEnd``, normalize using the discharges observed between
        ``flowNormStart`` and ``flowNormEnd``.

    Raises:
        ValueError: If Daily is missing or a break date falls outside it.
    """
    daily = elist["Daily"]
    if daily is None or daily.empty:
        raise ValueError("eList has no Daily data.")

    dates = pd.to_datetime(daily["Date"])
    q_start, q_end = dates.iloc[0], dates.iloc[-1]
    start = _as_date(surface_start) if surface_start is not None else q_start
    end = _as_date(surface_end) if surface_end is not None else q_end

    if flow_break_date is None:
        if window_side <= 0:
            return pd.DataFrame({
                "flowNormStart": [q_start], "flowNormEnd": [q_end],
                "flowStart": [start], "flowEnd": [end],
            })
        return _sliding_segments(start, end, q_start, q_end, window_side)

    break_date = _as_date(flow_break_date)
    if not (q_start < break_date < q_end):
        raise ValueError(
            f"flow_break_date {break_date.date()} is outside the discharge "
            f"record ({q_start.date()} to {q_end.date()})."
        )
    after_break = break_date + pd.Timedelta(days=1)

    if window_side <= 0:
        return pd.DataFrame({
            "flowNormStart": [q_start, after_break],
            "flowNormEnd": [break_date, q_end],
            "flowStart": [start, after_break],
            "flowEnd": [break_date, end],
        })

    first = _sliding_segments(start, break_date, q_start, break_date, window_side)
    second = _sliding_segments(after_break, end, after_break, q_end, window_side)
    return pd.concat([first, second], ignore_index=True)


def flex_fn(
    elist: dict,
    date_info: pd.DataFrame | None = None,
    window_side: int = 0,
    flow_break_date=None,
) -> pd.DataFrame:
    """Flow-normalize each stretch of the record against its own flows.

    Mirrors R ``flexFN()``.

    Args:
        elist: eList dict with fitted ``surfaces`` and grid axes.
        date_info: Segment table from :func:`make_date_info`. Built from
            ``window_side`` and ``flow_break_date`` if not given.
        window_side: Passed to :func:`make_date_info` when ``date_info`` is
            None. Default 0, standard flow normalization.
        flow_break_date: Passed to :func:`make_date_info` when ``date_info``
            is None.

    Returns:
        A copy of Daily with ``ConcDay``, ``FluxDay``, ``FNConc`` and
        ``FNFlux``. ``ConcDay`` and ``FluxDay`` do not depend on the
        normalization and are the same in every segment; only the
        flow-normalized pair changes.

        Days outside every segment get NaN, as in R.

    Raises:
        ValueError: If the surface has not been fitted.
    """
    if elist.get("surfaces") is None:
        raise ValueError("eList surfaces is None - fit the surface first.")

    if date_info is None:
        date_info = make_date_info(
            elist, window_side=window_side, flow_break_date=flow_break_date
        )

    daily = elist["Daily"].copy()
    dates = pd.to_datetime(daily["Date"])

    for column in ("ConcDay", "FluxDay", "FNConc", "FNFlux"):
        daily[column] = np.nan

    for _, segment in date_info.iterrows():
        reference = elist["Daily"][
            (dates >= segment["flowNormStart"]) & (dates <= segment["flowNormEnd"])
        ]
        if reference.empty:
            continue

        normalized = est_daily_from_surfaces(elist, flow_norm_daily=reference)
        applies = ((dates >= segment["flowStart"]) & (dates <= segment["flowEnd"])).to_numpy()
        for column in ("ConcDay", "FluxDay", "FNConc", "FNFlux"):
            daily.loc[applies, column] = normalized.loc[applies, column].to_numpy()

    # As in R, days outside every segment are left NaN.
    return daily

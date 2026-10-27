"""
merge_report — join INFO, Daily, and Sample into an eList with validation.
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from pyegret.data.populate import merge_daily_q
from pyegret.elist import make_elist

__all__ = ["merge_report"]

# Columns that must be present in Daily before building an eList
_REQUIRED_DAILY_COLS = {
    "Date", "Q", "LogQ", "DecYear", "Month", "Day",
    "MonthSeq", "Julian", "SinDY", "CosDY",
}


def merge_report(
    INFO: pd.DataFrame,
    Daily: pd.DataFrame,
    Sample: Optional[pd.DataFrame] = None,
) -> dict:
    """Validate and join site metadata, daily discharge, and WQ samples into an eList.

    Mirrors R ``mergeReport()``.

    Validation steps:
    - Checks that Daily contains all required EGRET columns.
    - If Sample is provided, merges daily Q / LogQ onto Sample by date
      (via ``merge_daily_q``).
    - Warns if any Sample dates fall outside the Daily date range.

    Args:
        INFO: 1-row DataFrame of site and parameter metadata.
        Daily: Daily discharge DataFrame (must have EGRET date columns).
        Sample: Optional WQ sample DataFrame.

    Returns:
        Validated :class:`~pyegret.model.egret_objects.eList`.

    Raises:
        ValueError: If Daily is missing required columns.
    """
    _validate_daily(Daily)

    if Sample is not None and len(Sample) > 0:
        Sample = merge_daily_q(Daily, Sample)
        _check_sample_date_range(Daily, Sample)

    return make_elist(INFO=INFO, Daily=Daily, Sample=Sample)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_daily(daily: pd.DataFrame) -> None:
    """Raise ValueError if *daily* is missing required EGRET columns."""
    missing = _REQUIRED_DAILY_COLS - set(daily.columns)
    if missing:
        raise ValueError(
            f"Daily DataFrame is missing required column(s): {sorted(missing)}. "
            "Run populate_date_columns() first."
        )


def _check_sample_date_range(daily: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Warn if any sample dates fall outside the daily discharge record."""
    d_min = daily["Date"].min()
    d_max = daily["Date"].max()
    s_min = sample["Date"].min()
    s_max = sample["Date"].max()

    if s_min < d_min:
        warnings.warn(
            f"Earliest sample date ({s_min.date()}) is before the start of the "
            f"Daily record ({d_min.date()}). Those samples will have no matched Q.",
            UserWarning,
            stacklevel=3,
        )
    if s_max > d_max:
        warnings.warn(
            f"Latest sample date ({s_max.date()}) is after the end of the "
            f"Daily record ({d_max.date()}). Those samples will have no matched Q.",
            UserWarning,
            stacklevel=3,
        )

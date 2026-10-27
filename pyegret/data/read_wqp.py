"""
Water Quality Portal (WQP / EPA STORET) sample retrieval.
Wraps ``dataretrieval.wqp.get_results()`` and converts to EGRET format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.data.populate import populate_date_columns, populate_sample_columns
from pyegret.utils.convert import as_dates, clean_text

__all__ = ["read_wqp_sample"]

# WQX v3 detection condition strings that indicate a censored (non-detect) result
_CENSORED_CONDITIONS = frozenset(
    {
        "Not Detected",
        "Present Below Quantification Limit",
        "Detected Not Quantified",
        "Not Reported",
        "Below Reporting Limit",
    }
)


def read_wqp_sample(
    site_id: str,
    characteristic_name: str,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Fetch water-quality samples from the Water Quality Portal (WQP).

    Mirrors R ``readWQPSample()``.
    Handles WQX v3 format (ActivityMeasure fields) and infers censoring
    from ``ResultDetectionConditionText``.

    Args:
        site_id: WQP site identifier (e.g. ``"USGS-01491000"``).
        characteristic_name: WQP characteristic name
            (e.g. ``"Nitrate"``).
        start_date: Start date ``"YYYY-MM-DD"`` or ``""``.
        end_date: End date ``"YYYY-MM-DD"`` or ``""``.

    Returns:
        Sample DataFrame with EGRET columns: ``Date``, ``ConcLow``,
        ``ConcHigh``, ``Uncen``, ``ConcAve``, ``LogConcLow``,
        ``LogConcHigh``, plus date columns.
    """
    try:
        from dataretrieval import wqp
    except ImportError as exc:
        raise ImportError(
            "dataretrieval is required for read_wqp_sample. "
            "Install with: pip install dataretrieval"
        ) from exc

    kwargs: dict = {
        "siteid": site_id,
        "characteristicName": characteristic_name,
    }
    if start_date:
        kwargs["startDateLo"] = start_date
    if end_date:
        kwargs["startDateHi"] = end_date

    df, _meta = wqp.get_results(**kwargs)
    df = df.reset_index(drop=True)

    result = _parse_wqp_results(df)
    result = populate_date_columns(result)
    result = populate_sample_columns(result)

    print(f"Retrieved {len(result)} sample records for site {site_id}")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_wqp_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw WQP results DataFrame to EGRET Sample format."""
    # WQX v3 date column
    date_col = next(
        (c for c in ("ActivityStartDate", "SampleDate", "startDateTime") if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"Cannot find date column in WQP results. Available: {list(df.columns)}"
        )
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Value column (WQX v3)
    val_col = next(
        (c for c in ("ResultMeasureValue", "ActivityMeasureValue", "result_va") if c in df.columns),
        None,
    )
    if val_col is None:
        raise ValueError(
            f"Cannot find result value column. Available: {list(df.columns)}"
        )
    values = pd.to_numeric(df[val_col], errors="coerce")

    # Detection condition column
    cond_col = next(
        (c for c in ("ResultDetectionConditionText", "ResultDetectionCondition") if c in df.columns),
        None,
    )
    if cond_col is not None:
        conditions = clean_text(df[cond_col])
        censored = conditions.isin(_CENSORED_CONDITIONS)
        censored = censored.to_numpy()
    else:
        censored = np.zeros(len(df), dtype=bool)

    uncen = np.where(censored, 0, 1)
    conc_high = np.asarray(values, dtype=float)
    conc_low = np.where(censored, 0.0, conc_high)

    result = pd.DataFrame(
        {
            "Date": np.asarray(dates),
            "ConcLow": conc_low,
            "ConcHigh": conc_high,
            "Uncen": uncen,
        }
    )
    result = result.dropna(subset=["Date", "ConcHigh"])
    result = result.reset_index(drop=True)
    result["Date"] = as_dates(result["Date"])
    return result

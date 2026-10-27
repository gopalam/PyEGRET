"""
USGS NWIS data retrieval through the legacy WaterServices endpoints.

.. deprecated::
   These functions call ``dataretrieval.nwis``, which reaches
   ``waterservices.usgs.gov``. USGS has said those endpoints will degrade and
   be superseded by ``api.waterdata.usgs.gov``: the groundwater levels service
   was decommissioned in November 2025, and while no date has been set for
   daily values or site metadata, the published guidance is to migrate now.

   Use :mod:`pyegret.data.read_waterdata` instead:

   ===========================  ==============================
   legacy                       replacement
   ===========================  ==============================
   ``read_nwis_daily``          ``read_waterdata_daily``
   ``read_nwis_sample``         ``read_waterdata_sample``
   ``read_nwis_info``           ``read_waterdata_info``
   ===========================  ==============================

   This module is kept working so existing scripts do not break.

Wraps the ``dataretrieval`` package and converts output to EGRET standard format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.data.populate import populate_date_columns, populate_sample_columns
from pyegret.utils.convert import as_dates, as_float_array, clean_text

__all__ = [
    "read_nwis_daily",
    "read_nwis_sample",
    "read_nwis_info",
]

# cfs → m³/s conversion factor
_CFS_TO_CMS = 0.028316847

# WQX v3 detection-condition strings that indicate a censored (non-detect) result.
# Mirrors the set used in read_wqp.py so both functions behave consistently.
_CENSORED_CONDITIONS = frozenset(
    {
        "Not Detected",
        "Present Below Quantification Limit",
        "Detected Not Quantified",
        "Not Reported",
        "Below Reporting Limit",
    }
)


def read_nwis_daily(
    site_id: str,
    parameter_cd: str = "00060",
    start_date: str = "",
    end_date: str = "",
    convert: bool = True,
) -> pd.DataFrame:
    """Fetch USGS NWIS mean-daily values and return an EGRET Daily DataFrame.

    Mirrors R ``readNWISDaily()``.

    Args:
        site_id: USGS site number (e.g. ``"01491000"``).
        parameter_cd: NWIS parameter code. Default ``"00060"`` (discharge, cfs).
        start_date: Start date string ``"YYYY-MM-DD"``, or ``""`` for full record.
        end_date: End date string ``"YYYY-MM-DD"``, or ``""`` for full record.
        convert: If True, multiply Q by 0.028316847 to convert cfs → m³/s.

    Returns:
        DataFrame with EGRET Daily columns: ``Date``, ``Q``, ``LogQ``,
        ``DecYear``, ``Month``, ``Day``, ``MonthSeq``, ``Julian``,
        ``SinDY``, ``CosDY``.
    """
    try:
        from dataretrieval import nwis
    except ImportError as exc:
        raise ImportError(
            "dataretrieval is required for read_nwis_daily. "
            "Install with: pip install dataretrieval"
        ) from exc

    kwargs: dict = {"sites": site_id, "parameterCd": parameter_cd}
    if start_date:
        kwargs["start"] = start_date
    if end_date:
        kwargs["end"] = end_date

    df, _meta = nwis.get_dv(**kwargs)
    df = df.reset_index()

    # Rename datetime index and discharge column to EGRET standard
    date_col = "datetime" if "datetime" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    # Find Q column: typically "{param_cd}_Mean" or similar
    q_col = _find_q_column(df, parameter_cd)
    df = df.rename(columns={q_col: "Q"})
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")

    if convert:
        df["Q"] = df["Q"] * _CFS_TO_CMS

    # Drop missing / non-positive discharge
    df = df.dropna(subset=["Q"])
    df = df[df["Q"] > 0].copy()

    df["LogQ"] = np.log(df["Q"])
    df = populate_date_columns(df)

    print(f"Retrieved {len(df)} daily values for site {site_id}")
    return df.reset_index(drop=True)

def _reformat_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to MM-DD-YYYY for WQP API."""
    if not date_str:
        return date_str
    parts = date_str.split("-")
    if len(parts) == 3 and len(parts[0]) == 4:
        return f"{parts[1]}-{parts[2]}-{parts[0]}"
    return date_str  # already in correct format or empty
def read_nwis_sample(
    site_id: str,
    parameter_cd: str,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Fetch USGS NWIS discrete water-quality samples via the Water Quality Portal.

    Mirrors R ``readNWISSample()``.

    Uses ``dataretrieval.wqp.get_results()`` with the NWIS parameter code
    and a ``USGS-`` prefixed site identifier.  The legacy
    ``nwis.get_qwdata()`` endpoint was retired by USGS in 2024 and no
    longer serves data.

    Censored values (WQX detection-condition strings such as
    ``"Not Detected"``) are encoded as
    ``Uncen=0``, ``ConcLow=0``, ``ConcHigh=reported_value``.

    Args:
        site_id: USGS site number (e.g. ``"01491000"``).  The ``USGS-``
            prefix is added automatically if absent.
        parameter_cd: NWIS parameter code (e.g. ``"00631"`` for nitrate).
        start_date: Start date ``"YYYY-MM-DD"`` or ``""``.
        end_date: End date ``"YYYY-MM-DD"`` or ``""``.

    Returns:
        Sample DataFrame with EGRET columns including ``ConcLow``,
        ``ConcHigh``, ``Uncen``, ``ConcAve``, ``LogConcLow``,
        ``LogConcHigh``, and date columns.
    """
    try:
        from dataretrieval import wqp
    except ImportError as exc:
        raise ImportError(
            "dataretrieval is required for read_nwis_sample. "
            "Install with: pip install dataretrieval"
        ) from exc

    # WQP requires the USGS- prefix for NWIS sites
    site_upper = site_id.upper()
    if site_upper.startswith("USGS-"):
        wqp_site_id = site_id
    else:
        wqp_site_id = f"USGS-{site_id}"

    kwargs: dict = {
        "siteid": wqp_site_id,
        "pCode": parameter_cd,          # ← WQP keyword for NWIS parameter codes
    }
    if start_date:
        kwargs["startDateLo"] = _reformat_date(start_date)
    if end_date:
        kwargs["startDateHi"] = _reformat_date(end_date)

    df, _meta = wqp.get_results(**kwargs)
    df = df.reset_index(drop=True)

    result = _parse_wqp_nwis(df)
    result = populate_date_columns(result)
    result = populate_sample_columns(result)

    print(f"Retrieved {len(result)} sample records for site {site_id}")
    return result.reset_index(drop=True)


def read_nwis_info(site_id: str, parameter_cd: str) -> pd.DataFrame:
    """Fetch site and parameter metadata from USGS NWIS.

    Mirrors R ``readNWISInfo()``.

    Args:
        site_id: USGS site number.
        parameter_cd: NWIS parameter code.

    Returns:
        1-row DataFrame with site metadata plus ``shortName``,
        ``paramShortName``, and ``param_cd`` columns.
    """
    try:
        from dataretrieval import nwis
    except ImportError as exc:
        raise ImportError(
            "dataretrieval is required for read_nwis_info."
        ) from exc

    site_info, _meta = nwis.get_info(sites=site_id)
    site_info = site_info.reset_index(drop=True)
    site_info = site_info.iloc[[0]].copy()

    # Friendly name columns used throughout PyEGRET
    if "station_nm" in site_info.columns:
        site_info["shortName"] = site_info["station_nm"].iloc[0]
    else:
        site_info["shortName"] = site_id

    # paramShortName: try to pull from parameter lookup
    param_name = _lookup_param_name(parameter_cd)
    site_info["paramShortName"] = param_name
    site_info["param_cd"] = parameter_cd

    return site_info


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_q_column(df: pd.DataFrame, parameter_cd: str) -> str:
    """Return the column name for the mean daily value of *parameter_cd*."""
    # Try common dataretrieval naming patterns
    candidates = [
        f"{parameter_cd}_Mean",
        f"p{parameter_cd}",
        parameter_cd,
    ]
    for name in candidates:
        if name in df.columns:
            return name
    # Last resort: first numeric column that isn't Date-related
    for col in df.columns:
        if col == "Date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    raise ValueError(
        f"Cannot find discharge column for parameter_cd={parameter_cd!r} "
        f"in columns: {list(df.columns)}"
    )


def _parse_wqp_nwis(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw WQP results DataFrame (NWIS site query) to EGRET Sample format.

    Handles WQX v3 column names returned by ``wqp.get_results()`` when
    queried with ``parameterCd``.  Detection-condition strings are mapped to
    the EGRET censoring convention (``Uncen=0``, ``ConcLow=0``).
    """
    # ── Date ─────────────────────────────────────────────────────────────────
    date_col = next(
        (c for c in ("ActivityStartDate", "SampleDate", "startDateTime") if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError(
            f"Cannot find date column in WQP results. Available: {list(df.columns)}"
        )
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # ── Concentration value ───────────────────────────────────────────────────
    val_col = next(
        (c for c in ("ResultMeasureValue", "ActivityMeasureValue", "result_va") if c in df.columns),
        None,
    )
    if val_col is None:
        raise ValueError(
            f"Cannot find result value column in WQP results. Available: {list(df.columns)}"
        )
    values = pd.to_numeric(df[val_col], errors="coerce")

    # ── Censoring ─────────────────────────────────────────────────────────────
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


    # For censored rows, ResultMeasureValue is NaN — fill ConcHigh 
    # from the detection limit column instead
    dl_col = "DetectionQuantitationLimitMeasure/MeasureValue"
    if dl_col in df.columns:
        dl_values = as_float_array(df[dl_col])
        conc_high = np.where(
            censored & np.isnan(conc_high),
            dl_values,
            conc_high
        )

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


def _lookup_param_name(parameter_cd: str) -> str:
    """Return a short human-readable name for common NWIS parameter codes."""
    _KNOWN: dict[str, str] = {
        "00060": "Discharge",
        "00631": "Nitrate",
        "00665": "Phosphorus",
        "00600": "Total Nitrogen",
        "00300": "Dissolved Oxygen",
        "00010": "Temperature",
        "00095": "Specific Conductance",
        "00400": "pH",
        "80154": "Suspended Sediment",
    }
    return _KNOWN.get(parameter_cd, f"Parameter {parameter_cd}")

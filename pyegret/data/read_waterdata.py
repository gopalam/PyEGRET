"""
USGS data retrieval through the modern Water Data APIs.

Replaces :mod:`pyegret.data.read_nwis`, which calls ``dataretrieval.nwis`` and so
reaches the legacy ``waterservices.usgs.gov`` endpoints. USGS has said those
will degrade and be superseded by ``api.waterdata.usgs.gov``; the groundwater
levels service was decommissioned in November 2025 and no date has been set for
daily values or site metadata, but the published guidance is to migrate now.

This module wraps ``dataretrieval.waterdata``:

=========================  ===============================================
this module                underlying call
=========================  ===============================================
``read_waterdata_daily``   ``waterdata.get_daily``
``read_waterdata_sample``  ``waterdata.get_samples``
``read_waterdata_info``    ``waterdata.get_monitoring_locations``
=========================  ===============================================

Column names differ between the legacy and modern services, and the Samples
service exposes several profiles with different schemas. Rather than assume a
layout, every mapping here searches a list of candidate names and raises an
error naming what actually came back. Pass ``raw=True`` to any function to get
the untouched response instead, which is the fastest way to diagnose a
mismatch: see ``diagnostics/check_waterdata.py``.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from pyegret.data.populate import populate_daily, populate_date_columns, populate_sample_columns
from pyegret.utils.convert import as_dates, as_float_array, clean_text

__all__ = [
    "read_waterdata_daily",
    "read_waterdata_sample",
    "read_waterdata_info",
    "usgs_location_id",
]

# cubic feet per second -> cubic metres per second
CFS_PER_CMS = 35.314667          # EGRET's qConst$cfs factor
CFS_TO_CMS = 1.0 / CFS_PER_CMS

# Statistic code for a daily mean.
DAILY_MEAN = "00003"

# Unit strings the daily service is known to use for discharge, lowercased.
_CFS_UNITS = frozenset({"ft3/s", "cfs", "ft^3/s", "cubic feet per second"})
_CMS_UNITS = frozenset({"m3/s", "cms", "m^3/s", "cubic meters per second"})

# Detection-condition text meaning "not detected at the reporting limit".
_CENSORED_CONDITIONS = frozenset({
    "not detected",
    "present below quantification limit",
    "detected not quantified",
    "below reporting limit",
    "below detection limit",
    "not present",
})


def usgs_location_id(site_id: str) -> str:
    """Return a monitoring location id in the modern ``USGS-#####`` form.

    Args:
        site_id: A bare NWIS site number (``"01491000"``) or an id that
            already carries an agency prefix (``"USGS-01491000"``).

    Returns:
        The prefixed identifier.
    """
    site_id = str(site_id)
    site_id = site_id.strip()
    return site_id if "-" in site_id else f"USGS-{site_id}"


def _time_interval(start_date: str, end_date: str) -> str | None:
    """Build an RFC 3339 interval for the OGC ``time`` parameter.

    Open-ended intervals use ``..`` on the missing side; two empty dates give
    None, meaning "no time filter".
    """
    start, end = (start_date or "").strip(), (end_date or "").strip()
    if not start and not end:
        return None
    return f"{start or '..'}/{end or '..'}"


def _simple_name(name) -> str:
    """Column name in lower case with underscores removed, for loose matching."""
    name = str(name)
    name = name.lower()
    return name.replace("_", "")


def _lower_names(columns) -> list:
    """Column names in lower case."""
    names = []
    for name in columns:
        name = str(name)
        names.append(name.lower())
    return names


def _first_column(frame: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    """Return the first candidate column present, else raise a useful error."""
    lookup = {}
    for col in frame.columns:
        name = str(col)
        lookup[name.lower()] = name
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(
        f"Could not find the {role} column in the response. Tried "
        f"{list(candidates)}; available columns are {list(frame.columns)}. "
        "Run diagnostics/check_waterdata.py to dump the raw response."
    )


def read_waterdata_daily(
    site_id: str,
    parameter_cd: str = "00060",
    start_date: str = "",
    end_date: str = "",
    convert: bool = True,
    statistic_id: str = DAILY_MEAN,
    raw: bool = False,
    adjust: bool = True,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Fetch mean daily values and return an EGRET Daily DataFrame.

    Modern-API replacement for
    :func:`~pyegret.data.read_nwis.read_nwis_daily`.

    Args:
        site_id: NWIS site number, with or without the ``USGS-`` prefix.
        parameter_cd: Parameter code. Default ``"00060"`` (discharge).
        start_date: ``"YYYY-MM-DD"`` or ``""`` for the whole record.
        end_date: ``"YYYY-MM-DD"`` or ``""`` for the whole record.
        convert: Convert discharge from ft³/s to m³/s. EGRET stores SI, and
            the conversion is applied only when the service reports the
            values in ft³/s. Default True.
        statistic_id: Statistic code. Default ``"00003"`` (daily mean).
        raw: Return the service response untouched, for diagnosis.
        adjust: Shift zero flows as EGRET does. Default True.
        verbose: Print EGRET's messages about zero and missing flows.
        **kwargs: Passed through to ``waterdata.get_daily``.

    Returns:
        DataFrame with EGRET Daily columns (see
        :func:`~pyegret.data.populate.populate_daily`). As in R
        ``readNWISDaily``, the record covers every calendar day from first to
        last; days the service did not report have Q = NaN.

    Raises:
        ImportError: If ``dataretrieval`` is not installed.
        ValueError: If the response has no recognisable time or value column.
    """
    waterdata = _import_waterdata()

    query = {
        "monitoring_location_id": usgs_location_id(site_id),
        "parameter_code": parameter_cd,
        "statistic_id": statistic_id,
        "skip_geometry": True,
    }
    interval = _time_interval(start_date, end_date)
    if interval:
        query["time"] = interval
    query.update(kwargs)

    frame = _as_frame(waterdata.get_daily(**query))
    if raw:
        return frame
    if frame.empty:
        raise ValueError(
            f"No daily values returned for {usgs_location_id(site_id)}, "
            f"parameter {parameter_cd}, statistic {statistic_id}."
        )

    time_col = _first_column(frame, ["time", "datetime", "date"], "date")
    value_col = _first_column(frame, ["value", "result", "00060_Mean"], "value")

    dates = as_dates(frame[time_col], utc=True)
    qualifier = None
    if "qualifier" in _lower_names(frame.columns):
        qualifier_col = _first_column(frame, ["qualifier"], "qualifier")
        qualifier = frame[qualifier_col].to_numpy()

    # Mirrors R readNWISDaily: values are divided by 35.314667 when they are
    # in ft3/s, then populateDaily fills the calendar and shifts zero flows.
    q_convert = 1.0
    if convert and _is_cubic_feet(frame, parameter_cd):
        q_convert = CFS_PER_CMS

    daily = populate_daily(dates, frame[value_col], q_convert=q_convert,
                           qualifier=qualifier, adjust=adjust, verbose=verbose)
    return daily


def read_waterdata_sample(
    site_id: str,
    parameter_cd: str,
    start_date: str = "",
    end_date: str = "",
    profile: str = "basicphyschem",
    raw: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Fetch discrete water-quality samples and return an EGRET Sample frame.

    Modern-API replacement for
    :func:`~pyegret.data.read_nwis.read_nwis_sample`, using the USGS Samples
    service that superseded the retired ``nwis.get_qwdata`` endpoint.

    Non-detects are encoded the EGRET way: ``Uncen=0``, ``ConcLow=0``,
    ``ConcHigh`` set to the reporting limit.

    Args:
        site_id: NWIS site number, with or without the ``USGS-`` prefix.
        parameter_cd: NWIS parameter code, e.g. ``"00631"`` for nitrate.
        start_date: ``"YYYY-MM-DD"`` or ``""``.
        end_date: ``"YYYY-MM-DD"`` or ``""``.
        profile: Samples results profile. Default ``"basicphyschem"``;
            ``"fullphyschem"`` returns more columns.
        raw: Return the service response untouched, for diagnosis.
        **kwargs: Passed through to ``waterdata.get_samples``.

    Returns:
        Sample DataFrame with ``Date``, ``ConcLow``, ``ConcHigh``, ``Uncen``,
        ``ConcAve``, ``LogConcLow``, ``LogConcHigh`` and the date columns.
        Discharge is not attached here; use
        :func:`~pyegret.data.merge_report.merge_report`.

    Raises:
        ImportError: If ``dataretrieval`` is not installed.
        ValueError: If the response has no recognisable date or result column.
    """
    waterdata = _import_waterdata()

    query = {
        "monitoring_location_id": usgs_location_id(site_id),
        "usgs_pcode": parameter_cd,
        "service": "results",
        "profile": profile,
    }
    if start_date:
        query["activity_start_date_lower"] = start_date
    if end_date:
        query["activity_start_date_upper"] = end_date
    query.update(kwargs)

    frame = _as_frame(waterdata.get_samples(**query))
    if raw:
        return frame
    if frame.empty:
        raise ValueError(
            f"No samples returned for {usgs_location_id(site_id)}, "
            f"parameter {parameter_cd}."
        )

    return _to_egret_sample(frame)


def read_waterdata_info(
    site_id: str,
    parameter_cd: str = "",
    raw: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Fetch monitoring-location metadata as a one-row INFO frame.

    Modern-API replacement for
    :func:`~pyegret.data.read_nwis.read_nwis_info`.

    Args:
        site_id: NWIS site number, with or without the ``USGS-`` prefix.
        parameter_cd: Parameter code, recorded on the INFO row for labelling.
        raw: Return the service response untouched, for diagnosis.
        **kwargs: Passed through to ``waterdata.get_monitoring_locations``.

    Returns:
        One-row DataFrame carrying the service's own columns plus
        ``shortName``, ``paramShortName``, ``param_cd`` and, where drainage
        area is reported, ``drainSqKm``.

    Raises:
        ImportError: If ``dataretrieval`` is not installed.
        ValueError: If the location is not found.
    """
    waterdata = _import_waterdata()

    frame = _as_frame(waterdata.get_monitoring_locations(
        monitoring_location_id=usgs_location_id(site_id),
        skip_geometry=True,
        **kwargs,
    ))
    if raw:
        return frame
    if frame.empty:
        raise ValueError(f"No monitoring location found for {usgs_location_id(site_id)}.")

    info = frame.reset_index(drop=True)
    info = info.iloc[[0]].copy()

    name_col = next(
        (c for c in ("monitoring_location_name", "station_nm", "name")
         if c in info.columns), None,
    )
    info["shortName"] = info[name_col].iloc[0] if name_col else str(site_id)

    area_col = next(
        (c for c in ("drainage_area", "drain_area_va", "contributing_drainage_area")
         if c in info.columns), None,
    )
    if area_col is not None:
        area = pd.to_numeric(info[area_col], errors="coerce").iloc[0]
        if pd.notna(area):
            # The service reports drainage area in square miles.
            info["drainSqKm"] = float(area) * 2.58998811
        else:
            info["drainSqKm"] = np.nan
    else:
        info["drainSqKm"] = np.nan

    info["param_cd"] = parameter_cd
    info["paramShortName"] = _parameter_name(parameter_cd)
    return info


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _import_waterdata():
    """Import ``dataretrieval.waterdata``, with an actionable error."""
    try:
        from dataretrieval import waterdata
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The modern USGS Water Data API needs dataretrieval 1.3 or newer. "
            "Install with: pip install 'dataretrieval>=1.3'"
        ) from exc
    return waterdata


def _as_frame(result) -> pd.DataFrame:
    """Normalise a dataretrieval return value to a DataFrame.

    Some calls return ``(frame, metadata)``; others return the frame alone.
    """
    if isinstance(result, tuple):
        result = result[0]
    if result is None:
        return pd.DataFrame()
    frame = pd.DataFrame(result)
    return frame.reset_index(drop=True) if not frame.empty else frame


def _is_cubic_feet(frame: pd.DataFrame, parameter_cd: str) -> bool:
    """Decide whether the returned discharge is in ft³/s.

    Trusts the service's own unit string when it gives one, and otherwise
    falls back to the parameter code: 00060 is defined in ft³/s.
    """
    unit_cols = []
    for col in frame.columns:
        if "unit" in _simple_name(col):
            unit_cols.append(col)
    for col in unit_cols:
        present = frame[col].dropna()
        units = set(clean_text(present.unique(), lower=True))
        if units & _CMS_UNITS:
            return False
        if units & _CFS_UNITS:
            return True
        if units:
            warnings.warn(
                f"Unrecognised discharge unit {sorted(units)!r}; assuming "
                f"{'ft3/s' if parameter_cd == '00060' else 'no conversion'}. "
                "Check the values before using them.",
                RuntimeWarning,
                stacklevel=3,
            )
            break
    return parameter_cd == "00060"


def _to_egret_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a Samples-service results frame to EGRET Sample format."""
    date_col = _first_column(
        frame,
        ["activity_start_date", "ActivityStartDate", "activityStartDate",
         "activity_start_date_time", "date"],
        "sample date",
    )
    value_col = _first_column(
        frame,
        ["result_measure_value", "ResultMeasureValue", "resultMeasureValue",
         "result_value", "value"],
        "result value",
    )

    dates = as_dates(frame[date_col])
    conc_high = as_float_array(frame[value_col])

    # Censoring: a detection-condition string marks a non-detect, and the
    # reporting limit then stands in for the upper bound.
    condition_col = None
    for col in frame.columns:
        if "detectioncondition" in _simple_name(col):
            condition_col = col
            break
    if condition_col is not None:
        text = clean_text(frame[condition_col], lower=True)
        listed = text.isin(_CENSORED_CONDITIONS)
        worded = text.str.contains("not detected|below", regex=True) & (text != "")
        censored = listed.to_numpy() | worded.to_numpy()
    else:
        censored = np.zeros(len(frame), dtype=bool)

    limit_col = None
    for col in frame.columns:
        name = _simple_name(col)
        if "detectionlimit" in name and "value" in name:
            limit_col = col
            break
    if limit_col is not None:
        limits = as_float_array(frame[limit_col])
        needs_limit = censored & ~np.isfinite(conc_high)
        conc_high = np.where(needs_limit, limits, conc_high)

    sample = pd.DataFrame({
        "Date": dates.to_numpy(),
        "ConcLow": np.where(censored, 0.0, conc_high),
        "ConcHigh": conc_high,
        "Uncen": np.where(censored, 0, 1),
    })
    sample = sample.dropna(subset=["Date", "ConcHigh"])
    sample = sample.sort_values("Date")
    sample = sample.reset_index(drop=True)

    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)
    return sample


def _parameter_name(parameter_cd: str) -> str:
    """Short human-readable name for common NWIS parameter codes."""
    known = {
        "00060": "Discharge",
        "00631": "Nitrate plus nitrite",
        "00665": "Total phosphorus",
        "00600": "Total nitrogen",
        "00300": "Dissolved oxygen",
        "00010": "Temperature",
        "00095": "Specific conductance",
        "00400": "pH",
        "80154": "Suspended sediment",
    }
    return known.get(parameter_cd, f"Parameter {parameter_cd}" if parameter_cd else "")

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

from pyegret.data.compress_data import compress_data, remove_zeros
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

# Detection-condition text that marks a non-detect. Mirrors R processQWData():
# a result is censored ("<") when its ResultDetectionConditionText contains
# any of these, ignoring case.
NON_DETECT_TEXT = (
    "NON-DETECT",
    "NON DETECT",
    "NOT DETECTED",
    "DETECTED NOT QUANTIFIED",
    "BELOW QUANTIFICATION LIMIT",
)

# The Samples service's column names, and the Water Quality Portal names that
# EGRET's code uses for them. Mirrors R ``conversion_names`` in readNWISSample.R.
SAMPLES_LEGACY_NAMES = {
    "Result_ResultDetectionCondition": "ResultDetectionConditionText",
    "Result_Measure": "ResultMeasureValue",
    "DetectionLimit_MeasureA": "DetectionQuantitationLimitMeasure.MeasureValue",
    "Result_Characteristic": "CharacteristicName",
    "Activity_StartDate": "ActivityStartDate",
    "Activity_StartDateTime": "ActivityStartDateTime",
    "USGSpcode": "USGSPCode",
    "Activity_TypeCode": "ActivityTypeCode",
    "Activity_Media": "ActivityMediaName",
    "Activity_MediaSubdivision": "ActivityMediaSubdivisionName",
    "SampleCollectionMethod_Name": "SampleCollectionMethod.MethodName",
    "ResultAnalyticalMethod_Identifier": "ResultAnalyticalMethod.MethodIdentifier",
    "Result_SampleFraction": "ResultSampleFractionText",
    "Result_MeasureStatusIdentifier": "ResultStatusIdentifier",
    "Result_MeasureType": "ResultValueTypeName",
}

# Columns R's processQWData() keeps, as (new name, source column).
SAMPLE_DESCRIPTION_COLUMNS = (
    ("CharacteristicName", "CharacteristicName"),
    ("USGSPCode", "USGSPCode"),
    ("ActivityStartDateTime", "ActivityStartDateTime"),
    ("ActivityTypeCode", "ActivityTypeCode"),
    ("ActivityMediaName", "ActivityMediaName"),
    ("ActivityMediaSubdivision", "ActivityMediaSubdivisionName"),
    ("SampleCollectionMethod", "SampleCollectionMethod.MethodName"),
    ("ResultAnalyticalMethod", "ResultAnalyticalMethod.MethodIdentifier"),
    ("ResultSampleFraction", "ResultSampleFractionText"),
    ("ResultStatusIdentifier", "ResultStatusIdentifier"),
    ("ResultValueTypeName", "ResultValueTypeName"),
)

# The characteristics processQWData() reports when they vary between samples.
SAMPLE_CHECK_COLUMNS = (
    "USGSPCode",
    "CharacteristicName",
    "ActivityTypeCode",
    "ActivityMediaName",
    "ActivityMediaSubdivision",
    "SampleCollectionMethod",
    "ResultAnalyticalMethod",
    "ResultSampleFraction",
)

# The Sample columns R puts first, in R's order.
SAMPLE_CORE_COLUMNS = (
    "Date", "ConcLow", "ConcHigh", "Uncen", "ConcAve", "Julian", "Month", "Day",
    "DecYear", "waterYear", "MonthSeq", "SinDY", "CosDY",
)


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

    Mirrors R ``readNWISDaily()`` in EGRET 3.0.12, which reads the USGS
    Water Data API. Replaces :func:`~pyegret.data.read_nwis.read_nwis_daily`.

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
    profile: str = "fullphyschem",
    raw: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Fetch discrete water-quality samples and return an EGRET Sample frame.

    Mirrors R ``readNWISSample()`` in EGRET 3.0.12, which reads the USGS
    Samples service (it replaced the retired ``nwis.get_qwdata`` endpoint).
    The steps are R's: rename the service's columns to the Water Quality
    Portal names EGRET uses, mark non-detects (``processQWData``), turn each
    result into censoring bounds (``compressData``), drop results reported as
    zero (``remove_zeros``), and add the date columns
    (``populateSampleColumns``).

    Non-detects are encoded the EGRET way: ``Uncen=0``, ``ConcLow=0`` and
    ``ConcHigh`` the reporting limit.

    As in R, **every result is kept**, including several on the same day and
    results whose activity type or medium differs from the rest; when those
    vary, a message lists the values. The descriptive columns
    (``ActivityTypeCode``, ``ActivityMediaName``, ``ActivityStartDateTime``
    and others) are carried in the returned frame so the record can be
    inspected and filtered.

    Args:
        site_id: NWIS site number, with or without the ``USGS-`` prefix.
        parameter_cd: A five-digit USGS parameter code, e.g. ``"00631"`` for
            nitrate plus nitrite, or else a characteristic name.
        start_date: ``"YYYY-MM-DD"`` or ``""``.
        end_date: ``"YYYY-MM-DD"`` or ``""``.
        profile: Samples results profile. Default ``"fullphyschem"``, as in R.
        raw: Return the service response untouched, for diagnosis.
        verbose: Print R's messages: the number of results and any sample
            characteristics that vary. Default True.
        **kwargs: Passed through to ``waterdata.get_samples``.

    Returns:
        Sample DataFrame sorted by date, with ``Date``, ``ConcLow``,
        ``ConcHigh``, ``Uncen``, ``ConcAve``, ``LogConcLow``, ``LogConcHigh``,
        the date columns, and the descriptive columns above. Discharge is not
        attached here; use :func:`~pyegret.data.merge_report.merge_report`.

    Raises:
        ImportError: If ``dataretrieval`` is not installed.
        ValueError: If nothing is returned, or the response has no sample
            date or result column.
    """
    waterdata = _import_waterdata()

    query = {
        "monitoring_location_id": usgs_location_id(site_id),
        "service": "results",
        "profile": profile,
    }
    # As in R: a five-digit number is a parameter code, anything else a
    # characteristic name.
    code = str(parameter_cd)
    if len(code) == 5 and code.isdigit():
        query["usgs_pcode"] = code
    else:
        query["characteristic"] = code
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

    return _to_egret_sample(frame, verbose=verbose)


def read_waterdata_info(
    site_id: str,
    parameter_cd: str = "",
    raw: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Fetch monitoring-location metadata as a one-row INFO frame.

    Mirrors R ``readNWISInfo()`` in EGRET 3.0.12, run non-interactively.
    Replaces :func:`~pyegret.data.read_nwis.read_nwis_info`.

    Args:
        site_id: NWIS site number, with or without the ``USGS-`` prefix.
        parameter_cd: Parameter code, recorded on the INFO row for labelling.
        raw: Return the service response untouched, for diagnosis.
        **kwargs: Passed through to ``waterdata.get_monitoring_locations``.

    Returns:
        One-row DataFrame carrying the service's own columns plus
        ``shortName`` and ``station_nm`` (the site name), ``drainSqKm``
        (NaN where no drainage area is reported), ``param_cd``,
        ``paramShortName``, ``param.nm``, ``param.units`` and ``paramNumber``
        (from the USGS parameter-code table, as in R), and ``paStart`` = 10,
        ``paLong`` = 12.

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

    # R's readNWISInfo labels the site by its name, as station_nm.
    info["station_nm"] = info["shortName"]

    # Mirrors R readNWISInfo: the parameter's name and units come from the
    # USGS parameter-code table. If that lookup fails, fall back to a short
    # built-in list of common codes.
    info["param_cd"] = parameter_cd
    info["paramShortName"] = _parameter_name(parameter_cd)
    if parameter_cd:
        parameter = _parameter_record(waterdata, parameter_cd)
        if parameter is not None:
            info["param.nm"] = parameter["parameter_name"]
            info["param.units"] = parameter["unit_of_measure"]
            info["paramShortName"] = parameter["parameter_name"]
            info["paramNumber"] = parameter["parameter_code"]
            units = str(parameter["unit_of_measure"])
            units = units.upper()
            if parameter_cd not in ("00060", "00065") and "MG/L" not in units:
                warnings.warn(
                    f"Parameter {parameter_cd} is reported in {parameter['unit_of_measure']!r}. "
                    "EGRET computes flux from concentration in mg/L, so flux "
                    "results will be wrong for other units.",
                    UserWarning, stacklevel=2)

    # The period of analysis, as R sets it: the water year.
    info["paStart"] = 10
    info["paLong"] = 12
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


def _column_or_missing(frame: pd.DataFrame, name: str) -> pd.Series:
    """The named column, or a column of missing values if it is absent."""
    if name in frame.columns:
        column = frame[name]
        return column.reset_index(drop=True)
    return pd.Series([np.nan] * len(frame))


def _process_qw_data(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Mark non-detects and choose each result's value. Mirrors R ``processQWData()``.

    ``data`` has the Water Quality Portal column names (see
    ``SAMPLES_LEGACY_NAMES``). A non-detect gets the qualifier ``"<"`` and its
    detection limit as the value; anything else keeps its measured value.
    """
    for needed in ("ActivityStartDate", "ResultMeasureValue"):
        if needed not in data.columns:
            raise ValueError(
                f"The Samples response has no {needed!r} column. Available "
                f"columns are {list(data.columns)}. Run "
                "diagnostics/check_waterdata.py to dump the raw response."
            )

    detect_text = clean_text(_column_or_missing(data, "ResultDetectionConditionText"))
    detect_text = detect_text.str.upper()
    censored = np.zeros(len(data), dtype=bool)
    for text in NON_DETECT_TEXT:
        found = detect_text.str.contains(text, regex=False)
        censored = censored | found.to_numpy()

    measured = as_float_array(_column_or_missing(data, "ResultMeasureValue"))
    limit = as_float_array(
        _column_or_missing(data, "DetectionQuantitationLimitMeasure.MeasureValue"))

    processed = pd.DataFrame({
        "dateTime": as_dates(_column_or_missing(data, "ActivityStartDate")),
        "qualifier": np.where(censored, "<", ""),
        "value": np.where(censored, limit, measured),
    })
    for new_name, source in SAMPLE_DESCRIPTION_COLUMNS:
        processed[new_name] = _column_or_missing(data, source)

    n_retrieved = len(processed)
    has_date = processed["dateTime"].notna()
    processed = processed[has_date]
    processed = processed.reset_index(drop=True)
    if verbose:
        print(f"{n_retrieved} samples retrieved.")
    if len(processed) != n_retrieved:
        warnings.warn(f"{n_retrieved - len(processed)} samples removed because "
                      "date is missing.", UserWarning, stacklevel=3)

    if verbose:
        varying = []
        for column in SAMPLE_CHECK_COLUMNS:
            values = pd.unique(processed[column])
            if len(values) > 1:
                varying.append((column, values))
        if varying:
            print("Multiple values for some sample characteristics:")
            for column, values in varying:
                listed = "', '".join(str(value) for value in values)
                print(f"{column}: '{listed}'")
    return processed


def _to_egret_sample(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Convert a Samples-service results frame to EGRET Sample format.

    Mirrors the body of R ``readNWISSample()``: rename, ``processQWData``,
    ``compressData``, ``remove_zeros``, ``populateSampleColumns``.
    """
    renamed = frame.rename(columns=SAMPLES_LEGACY_NAMES)
    processed = _process_qw_data(renamed, verbose=verbose)

    first_three = ["dateTime", "qualifier", "value"]
    compressed = compress_data(processed[first_three], verbose=verbose)
    described = processed.drop(columns=first_three)
    combined = pd.concat([compressed, described], axis=1)
    combined = remove_zeros(combined, verbose=verbose)

    sample = pd.DataFrame({
        "Date": combined["dateTime"].to_numpy(),
        "ConcLow": combined["ConcLow"].to_numpy(dtype=float),
        "ConcHigh": combined["ConcHigh"].to_numpy(dtype=float),
        "Uncen": combined["Uncen"].to_numpy(dtype=int),
    })
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)
    for new_name, _ in SAMPLE_DESCRIPTION_COLUMNS:
        sample[new_name] = combined[new_name].to_numpy()

    first = [name for name in SAMPLE_CORE_COLUMNS if name in sample.columns]
    rest = [name for name in sample.columns if name not in first]
    sample = sample[first + rest]
    # A stable sort keeps several results on one day in the service's order.
    sample = sample.sort_values("Date", kind="stable")
    sample = sample.reset_index(drop=True)
    return sample


def _parameter_record(waterdata, parameter_cd: str):
    """The parameter-code table's entry for one code, or None if unavailable."""
    try:
        result = waterdata.get_reference_table("parameter-codes",
                                               query={"id": parameter_cd})
    except Exception:  # an older dataretrieval, a stub, or a network problem
        return None
    table = _as_frame(result)
    if table.empty:
        return None
    for column in ("parameter_name", "unit_of_measure", "parameter_code"):
        if column not in table.columns:
            return None
    row = table.iloc[0]
    return {
        "parameter_name": row["parameter_name"],
        "unit_of_measure": row["unit_of_measure"],
        "parameter_code": row["parameter_code"],
    }


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

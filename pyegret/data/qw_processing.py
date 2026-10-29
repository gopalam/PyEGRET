"""
Turning water-quality results from USGS or the Water Quality Portal into a Sample.

Shared by :func:`~pyegret.data.read_waterdata.read_waterdata_sample` (R
``readNWISSample``) and :func:`~pyegret.data.read_wqp.read_wqp_sample` (R
``readWQPSample``), which in EGRET 3.0.12 run the same steps once their
columns carry the Water Quality Portal names:

1. ``processQWData``: mark non-detects and choose each result's value;
2. ``compressData``: turn the remark and value into censoring bounds;
3. ``remove_zeros``: drop results reported as zero;
4. ``populateSampleColumns``: add the date and derived columns.

Column names may use R's dots (``DetectionQuantitationLimitMeasure.MeasureValue``)
or the slashes that Python's dataretrieval returns
(``DetectionQuantitationLimitMeasure/MeasureValue``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pyegret.data.compress_data import compress_data, remove_zeros
from pyegret.data.populate import populate_date_columns, populate_sample_columns
from pyegret.utils.convert import as_dates, as_float_array, clean_text

__all__ = [
    "process_wqp_results",
    "wqp_results_to_sample",
    "NON_DETECT_TEXT",
]

# Detection-condition text that marks a non-detect, matched anywhere in the
# text and ignoring case. The first five are R processQWData()'s. The last two
# are not in R, which treats such results as having no value and drops them;
# agencies do use them (Washington's county programs report "Below Detection
# Limit" through the Water Quality Portal), so PyEGRET censors them instead.
NON_DETECT_TEXT = (
    "NON-DETECT",
    "NON DETECT",
    "NOT DETECTED",
    "DETECTED NOT QUANTIFIED",
    "BELOW QUANTIFICATION LIMIT",
    "BELOW DETECTION LIMIT",
    "BELOW REPORTING LIMIT",
)

# Columns processQWData() keeps, as (new name, Water Quality Portal column).
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

LIMIT_COLUMN = "DetectionQuantitationLimitMeasure.MeasureValue"


def _dotted_names(data: pd.DataFrame) -> pd.DataFrame:
    """Rename ``A/B`` columns to R's ``A.B``."""
    names = {}
    for column in data.columns:
        name = str(column)
        if "/" in name:
            names[column] = name.replace("/", ".")
    if not names:
        return data
    return data.rename(columns=names)


def _column_or_missing(frame: pd.DataFrame, name: str) -> pd.Series:
    """The named column, or a column of missing values if it is absent."""
    if name in frame.columns:
        column = frame[name]
        return column.reset_index(drop=True)
    return pd.Series([np.nan] * len(frame))


def process_wqp_results(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Mark non-detects and choose each result's value. Mirrors R ``processQWData()``.

    A result whose ``ResultDetectionConditionText`` contains one of
    ``NON_DETECT_TEXT`` gets the qualifier ``"<"`` and its detection limit as
    its value; every other result keeps its measured value. Results without a
    date are dropped with a warning. A non-detect with no detection limit in
    the record cannot be used; it is counted in a warning here and dropped
    later by :func:`~pyegret.data.compress_data.remove_zeros`, as in R.

    Args:
        data: Results with Water Quality Portal column names
            (``ActivityStartDate``, ``ResultMeasureValue``,
            ``ResultDetectionConditionText``,
            ``DetectionQuantitationLimitMeasure.MeasureValue``, ...), with dots
            or slashes.
        verbose: Print R's messages: the number of results, and any sample
            characteristics that take more than one value. Default True.

    Returns:
        DataFrame with ``dateTime``, ``qualifier`` and ``value``, then the
        descriptive columns R keeps (``CharacteristicName``,
        ``ActivityTypeCode``, ``ActivityMediaName``, ``ResultSampleFraction``
        and others).

    Raises:
        ValueError: If there is no date or result column.
    """
    data = _dotted_names(data)
    for needed in ("ActivityStartDate", "ResultMeasureValue"):
        if needed not in data.columns:
            raise ValueError(
                f"The results have no {needed!r} column. Available columns are "
                f"{list(data.columns)}. Run diagnostics/check_waterdata.py to dump "
                "the raw response."
            )

    detect_text = clean_text(_column_or_missing(data, "ResultDetectionConditionText"))
    detect_text = detect_text.str.upper()
    censored = np.zeros(len(data), dtype=bool)
    for text in NON_DETECT_TEXT:
        found = detect_text.str.contains(text, regex=False)
        censored = censored | found.to_numpy()

    measured = as_float_array(_column_or_missing(data, "ResultMeasureValue"))
    limit = as_float_array(_column_or_missing(data, LIMIT_COLUMN))

    processed = pd.DataFrame({
        "dateTime": as_dates(_column_or_missing(data, "ActivityStartDate")),
        "qualifier": np.where(censored, "<", ""),
        "value": np.where(censored, limit, measured),
    })
    for new_name, source in SAMPLE_DESCRIPTION_COLUMNS:
        processed[new_name] = _column_or_missing(data, source)

    n_no_limit = int(np.sum(censored & ~np.isfinite(limit)))
    if n_no_limit > 0:
        warnings.warn(
            f"{n_no_limit} results are marked as not detected but give no detection "
            "limit, so they cannot be used and will be dropped. Ask the data "
            "provider for the reporting limit if they matter.",
            UserWarning, stacklevel=2)

    n_retrieved = len(processed)
    has_date = processed["dateTime"].notna()
    processed = processed[has_date]
    processed = processed.reset_index(drop=True)
    if verbose:
        print(f"{n_retrieved} samples retrieved.")
    if len(processed) != n_retrieved:
        warnings.warn(f"{n_retrieved - len(processed)} samples removed because "
                      "date is missing.", UserWarning, stacklevel=2)

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


def wqp_results_to_sample(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Turn Water Quality Portal results into an EGRET Sample.

    Mirrors the body shared by R ``readNWISSample()`` and ``readWQPSample()``:
    ``processQWData``, ``compressData``, ``remove_zeros`` and
    ``populateSampleColumns``. Every result is kept, including several on one
    day, and the descriptive columns are carried so the record can be
    inspected and filtered.

    Args:
        data: Results with Water Quality Portal column names (dots or slashes).
        verbose: Print R's messages. Default True.

    Returns:
        Sample DataFrame sorted by date: ``Date``, ``ConcLow``, ``ConcHigh``,
        ``Uncen``, ``ConcAve``, the date columns, ``LogConcLow``,
        ``LogConcHigh`` and the descriptive columns.
    """
    processed = process_wqp_results(data, verbose=verbose)

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
    # A stable sort keeps several results on one day in the order received.
    sample = sample.sort_values("Date", kind="stable")
    sample = sample.reset_index(drop=True)
    return sample

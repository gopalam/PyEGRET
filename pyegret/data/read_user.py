"""
Readers for user-supplied daily discharge and water-quality files.

Mirrors R ``readUserDaily()``, ``readUserSample()`` and ``readDataFromFile()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.data.compress_data import compress_data, remove_zeros
from pyegret.data.populate import (
    populate_daily,
    populate_date_columns,
    populate_sample_columns,
)
from pyegret.utils.convert import as_dates
from pyegret.utils.units import get_q_unit

__all__ = [
    "read_data_from_file",
    "read_user_daily",
    "read_user_sample",
]


def read_data_from_file(
    file_path: str,
    separator: str = ",",
    has_header: bool = True,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Read a delimited file whose first column is a date.

    Mirrors R ``readDataFromFile()``. Every column is read as text; lines
    starting with ``#`` are comments. ISO dates (``2001-03-15``) are
    recognised directly; anything else is parsed with ``date_format``
    (R's default is ``"%m/%d/%Y"``, used here when the ISO parse fails).

    Args:
        file_path: Path to the file.
        separator: Column delimiter. Default ``","``.
        has_header: First line holds column names. Default True.
        date_format: strftime format of the date column, if not ISO.

    Returns:
        DataFrame with the first column parsed as dates.

    Raises:
        ValueError: If any date cannot be parsed. EGRET cannot handle missing
            dates.
    """
    header = 0
    if not has_header:
        header = None
    data = pd.read_csv(file_path, sep=separator, header=header, dtype=str,
                       comment="#", skipinitialspace=True, keep_default_na=False)

    first = data.iloc[:, 0]
    if date_format is None:
        dates = pd.to_datetime(first, format="%Y-%m-%d", errors="coerce")
        if np.any(dates.isna()):
            dates = pd.to_datetime(first, format="%m/%d/%Y", errors="coerce")
    else:
        dates = pd.to_datetime(first, format=date_format, errors="coerce")
    if np.any(dates.isna()):
        raise ValueError(
            "Some dates are missing, EGRET functions cannot handle missing dates. "
            "Check the date format in the file and adjust the 'date_format' argument."
        )
    data[data.columns[0]] = as_dates(dates)
    return data


def read_user_daily(
    file_path: str,
    separator: str = ",",
    q_unit=1,
    has_header: bool = True,
    date_format: str | None = None,
    adjust: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a two-column daily discharge file into an EGRET Daily DataFrame.

    Mirrors R ``readUserDaily()``. The file has a date column then a
    discharge column; names do not matter, position does.

    **As in R, discharge is assumed to be in ft3/s unless told otherwise**
    (``q_unit=1``). Pass ``q_unit=2`` for m3/s.

    Args:
        file_path: Path to the file.
        separator: Column delimiter. Default ``","``.
        q_unit: Units of the discharge in the file: 1 cfs (default), 2 cms,
            3 thousand cfs, 4 thousand cms, or the EGRET name.
        has_header: First line holds column names. Default True.
        date_format: strftime format of the dates, if not ISO or m/d/Y.
        adjust: Shift zero flows as EGRET does. Default True.
        verbose: Print EGRET's messages. Default True.

    Returns:
        Daily DataFrame (``Date``, ``Q`` in m3/s, ``LogQ``, ``Q7``, ``Q30``
        and the date columns), with days that have no discharge removed.
    """
    data = read_data_from_file(file_path, separator, has_header, date_format)
    unit = get_q_unit(q_unit)
    if unit["name"] == "cfs" and verbose:
        print("The input discharge are assumed to be in cubic feet per second, "
              "if they are in cubic meters per second, then the call to "
              "read_user_daily should specify q_unit=2")

    daily = populate_daily(data.iloc[:, 0], data.iloc[:, 1],
                           q_convert=unit["factor"], adjust=adjust, verbose=verbose)
    daily = daily.dropna(subset=["Q"])
    daily = daily.reset_index(drop=True)
    return daily


def read_user_sample(
    file_path: str,
    separator: str = ",",
    has_header: bool = True,
    date_format: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a water-quality file into an EGRET Sample DataFrame.

    Mirrors R ``readUserSample()``. The file has a date column, then one or
    more pairs of (remark code, value) columns. Remark codes are blank for a
    detection, ``<`` for a value below the reporting limit and ``>`` for one
    above it. Several pairs are summed into one concentration (see
    :func:`~pyegret.data.compress_data.compress_data`). Rows reported as 0.0
    are deleted, with a warning, as in R.

    Args:
        file_path: Path to the file.
        separator: Column delimiter. Default ``","``.
        has_header: First line holds column names. Default True.
        date_format: strftime format of the dates, if not ISO or m/d/Y.
        verbose: Print EGRET's messages. Default True.

    Returns:
        Sample DataFrame sorted by date, with ``ConcLow``, ``ConcHigh``,
        ``Uncen``, ``ConcAve``, ``LogConcLow``, ``LogConcHigh`` and the date
        columns.
    """
    data = read_data_from_file(file_path, separator, has_header, date_format)
    data = compress_data(data, verbose=verbose)
    data = remove_zeros(data, verbose=verbose)

    sample = pd.DataFrame({
        "Date": data["dateTime"].to_numpy(),
        "ConcLow": data["ConcLow"].to_numpy(dtype=float),
        "ConcHigh": data["ConcHigh"].to_numpy(dtype=float),
        "Uncen": data["Uncen"].to_numpy(dtype=int),
    })
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)
    sample = sample.sort_values("Date")
    sample = sample.reset_index(drop=True)
    return sample

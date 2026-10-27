"""
Turning raw remark/value columns into censoring bounds, and cleaning a Sample.

Mirrors R ``compressData()``, ``remove_zeros()`` (internal in EGRET) and
``removeDuplicates()``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pyegret.data.quality import populate_concentrations
from pyegret.utils.convert import as_float_array, clean_text

__all__ = [
    "compress_data",
    "remove_zeros",
    "remove_duplicates",
]


def compress_data(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Combine remark/value column pairs into ConcLow, ConcHigh and Uncen.

    Mirrors R ``compressData()``. The first column is the date; the columns
    after it come in pairs, a remark code (``""``, ``"<"`` or ``">"``) then a
    value. With more than one pair - for example nitrate and nitrite reported
    separately - the bounds of the pairs are *summed*, which is how EGRET
    builds a total from its parts::

        date,       rn, nitrate, rd, nitrite
        2001-03-01, ,   1.2,     <,  0.01

    gives ConcLow = 1.2 and ConcHigh = 1.21 (censored, Uncen = 0).

    A missing value counts as 0, as in R.

    Args:
        data: DataFrame of date, then code/value pairs.
        verbose: Kept for parity with R; no effect.

    Returns:
        The input with its first three columns renamed ``dateTime``, ``code``
        and ``value``, and ``ConcLow``, ``ConcHigh`` and ``Uncen`` added.
        ``Uncen`` is 1 only where ``ConcLow == ConcHigh``.
    """
    data = data.copy()
    data = data.reset_index(drop=True)
    n_rows = len(data)
    n_pairs = (data.shape[1] - 1) // 2

    low_total = np.zeros(n_rows)
    high_total = np.zeros(n_rows)
    for i in range(n_pairs):
        codes = clean_text(data.iloc[:, 1 + 2 * i])
        values = as_float_array(data.iloc[:, 2 + 2 * i])
        values = np.where(np.isnan(values), 0.0, values)
        bounds = populate_concentrations(values, codes)
        low_total = low_total + bounds["ConcLow"].to_numpy(dtype=float)
        high_total = high_total + bounds["ConcHigh"].to_numpy(dtype=float)

    names = list(data.columns)
    names[0] = "dateTime"
    if n_pairs >= 1:
        names[1] = "code"
        names[2] = "value"
    data.columns = names

    data["ConcLow"] = low_total
    data["ConcHigh"] = high_total
    uncen = np.where(low_total == high_total, 1, 0)
    missing = np.isnan(low_total) | np.isnan(high_total)
    data["Uncen"] = np.where(missing, 0, uncen)
    return data


def remove_zeros(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop rows whose concentration was reported as exactly 0.

    Mirrors EGRET's internal ``remove_zeros()``, called by ``readUserSample``
    and ``readWQPSample``: a result of 0.0 cannot be interpreted (it is not a
    detection and has no reporting limit), so it is deleted with a warning.

    Args:
        data: Frame with ``ConcLow`` and ``ConcHigh``.
        verbose: Print the deleted rows. Default True.

    Returns:
        The frame without those rows, index reset.
    """
    is_zero = (data["ConcLow"] == 0) & (data["ConcHigh"] == 0)
    n_zero = int(np.sum(is_zero))
    if n_zero > 0:
        warnings.warn(
            f"Deleted {n_zero} rows of data because concentration was reported "
            "as 0.0, the program is unable to interpret that result and is "
            "therefore deleting it.",
            UserWarning,
            stacklevel=2,
        )
        if verbose:
            print("Deleted Rows:")
            print(data[is_zero])
    kept = data[~is_zero]
    kept = kept.reset_index(drop=True)
    return kept


def remove_duplicates(sample: pd.DataFrame) -> pd.DataFrame:
    """Drop samples that repeat an earlier sample's date and ConcHigh.

    Mirrors R ``removeDuplicates()``:
    ``Sample[!duplicated(Sample[c("DecYear", "ConcHigh")]), ]``.

    Args:
        sample: Sample DataFrame with ``DecYear`` (or ``Date``) and ``ConcHigh``.

    Returns:
        DataFrame with the repeats removed, index reset.
    """
    if "DecYear" in sample.columns:
        key = ["DecYear", "ConcHigh"]
    else:
        key = ["Date", "ConcHigh"]
    result = sample.drop_duplicates(subset=key)
    result = result.reset_index(drop=True)
    return result

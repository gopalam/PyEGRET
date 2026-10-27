"""
Monthly means of the daily WRTDS results. Mirrors R ``calculateMonthlyResults()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calculate_monthly_results"]


def calculate_monthly_results(elist: dict) -> pd.DataFrame:
    """Average the daily results within each calendar month of the record.

    Mirrors R ``calculateMonthlyResults()``. Months with 15 or fewer days of
    ``ConcDay`` estimates are dropped.

    Args:
        elist: eList dict whose Daily has ``ConcDay``, ``FluxDay``,
            ``FNConc`` and ``FNFlux`` (and ``GenConc`` / ``GenFlux`` after
            WRTDS-Kalman).

    Returns:
        DataFrame with ``Month``, ``Year``, ``nDays``, ``DecYear``, ``Q``,
        ``Conc``, [``GenConc``,] ``FNConc``, ``Flux``, [``GenFlux``,]
        ``FNFlux``, one row per month.
    """
    daily = elist["Daily"]
    month_seq = daily["MonthSeq"].to_numpy(dtype=int)
    months = np.unique(month_seq)
    kalman = "GenConc" in daily.columns and "GenFlux" in daily.columns

    names = ["Month", "Year", "nDays", "DecYear", "Q", "Conc"]
    if kalman:
        names = names + ["GenConc", "FNConc", "Flux", "GenFlux", "FNFlux"]
    else:
        names = names + ["FNConc", "Flux", "FNFlux"]
    source = {
        "DecYear": "DecYear", "Q": "Q", "Conc": "ConcDay", "Flux": "FluxDay",
        "FNConc": "FNConc", "FNFlux": "FNFlux", "GenConc": "GenConc",
        "GenFlux": "GenFlux", "Month": "Month",
    }
    values = {}
    for name in source:
        if source[name] in daily.columns:
            values[name] = daily[source[name]].to_numpy(dtype=float)

    out = {name: np.full(len(months), np.nan) for name in names}
    conc = values["Conc"]
    for k in range(len(months)):
        rows = month_seq == months[k]
        out["nDays"][k] = np.sum(~np.isnan(conc[rows]))
        for name in names:
            if name in ("nDays", "Year"):
                continue
            out[name][k] = np.mean(values[name][rows])
        out["Year"][k] = np.trunc(out["DecYear"][k])

    result = pd.DataFrame(out)
    result = result[result["nDays"] > 15]
    result = result.reset_index(drop=True)
    return result

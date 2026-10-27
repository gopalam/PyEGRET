"""
Non-parametric trend tests: Theil-Sen slope and Mann-Kendall, with the
Zhang prewhitening correction for serial correlation.

Streamflow series break the assumptions of ordinary regression twice over.
They are strongly skewed, so least squares chases the floods; and consecutive
years are correlated, so an ordinary significance test counts the same
information more than once and calls trends significant that are not.

Theil-Sen answers the first problem by taking the median of all pairwise
slopes, which ignores the magnitude of outliers entirely. Mann-Kendall answers
it for the test, ranking the data rather than using its values. Zhang's
iterative prewhitening answers the second: it estimates how much of the
year-to-year correlation is left after removing the trend, and discounts the
test accordingly.

References:
    Sen, P.K. (1968), *J. Am. Stat. Assoc.* 63, 1379-1389.
    Zhang, X., Vincent, L.A., Hogg, W.D., Niitsoo, A. (2000),
    *Atmosphere-Ocean* 38, 395-429 - the prewhitening procedure implemented
    in the R package ``zyp`` as ``zyp.zhang``, which EGRET's streamflow trend
    workflow calls.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "theil_sen_slope",
    "mann_kendall",
    "zhang_prewhitened_test",
]

# zyp.zhang: below this lag-1 autocorrelation the series is left alone.
_AUTOCORR_THRESHOLD = 0.05
_MAX_ITERATIONS = 500
_TREND_TOLERANCE = 0.001
_AUTOCORR_TOLERANCE = 0.0001


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median of the slopes between every pair of points.

    Args:
        x: Independent variable, length n.
        y: Dependent variable, length n.

    Returns:
        The Theil-Sen slope estimate, or NaN if fewer than two usable points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")

    i, j = np.triu_indices(x.size, k=1)
    dx = x[j] - x[i]
    usable = dx != 0
    if not usable.any():
        return float("nan")
    return float(np.median((y[j] - y[i])[usable] / dx[usable]))


def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation, matching R's ``acf`` (biased, n in the divisor)."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    n = finite.size
    if n < 3:
        return float("nan")
    centred = finite - finite.mean()
    denominator = float(np.sum(centred ** 2))
    if denominator == 0:
        return float("nan")
    return float(np.sum(centred[:-1] * centred[1:]) / denominator)


def mann_kendall(x: np.ndarray, y: np.ndarray) -> dict:
    """Mann-Kendall trend test with tie and continuity corrections.

    Args:
        x: Independent variable (typically year).
        y: Dependent variable.

    Returns:
        Dictionary with ``tau`` (Kendall's tau-b), ``S`` (the Mann-Kendall
        statistic), ``p_value`` (two-sided, normal approximation) and ``n``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3:
        return {"tau": np.nan, "S": np.nan, "p_value": np.nan, "n": n}

    order = np.argsort(x, kind="mergesort")
    y = y[order]

    i, j = np.triu_indices(n, k=1)
    s = float(np.sum(np.sign(y[j] - y[i])))

    def _tie_term(values):
        _, counts = np.unique(values, return_counts=True)
        tied = counts[counts > 1].astype(float)
        return float(np.sum(tied * (tied - 1) * (2 * tied + 5)))

    variance = (n * (n - 1) * (2 * n + 5) - _tie_term(y) - _tie_term(x[order])) / 18.0
    if variance <= 0:
        return {"tau": np.nan, "S": s, "p_value": np.nan, "n": n}

    # Continuity correction: shrink |S| by 1 before standardising.
    z = (s - np.sign(s)) / np.sqrt(variance) if s != 0 else 0.0
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))

    def _tie_pairs(values):
        _, counts = np.unique(values, return_counts=True)
        tied = counts[counts > 1].astype(float)
        return float(np.sum(tied * (tied - 1) / 2.0))

    total_pairs = n * (n - 1) / 2.0
    denominator = np.sqrt(
        (total_pairs - _tie_pairs(y)) * (total_pairs - _tie_pairs(x[order]))
    )
    tau = float(s / denominator) if denominator > 0 else np.nan

    return {"tau": tau, "S": s, "p_value": p_value, "n": n}


def zhang_prewhitened_test(y: np.ndarray, x: np.ndarray | None = None) -> dict:
    """Mann-Kendall test after Zhang's iterative prewhitening.

    Mirrors R ``zyp::zyp.zhang``, which EGRET's streamflow trend workflow uses
    for the p-values on the Quantile-Kendall plot.

    Serial correlation inflates the apparent significance of a trend: if this
    year's flow tells you something about next year's, consecutive
    observations are not independent evidence. The procedure estimates the
    trend, removes it, measures what autocorrelation remains, prewhitens the
    original series against it, and re-estimates - iterating until the trend
    and the autocorrelation both settle. The test is then run on the
    prewhitened series.

    Series with lag-1 autocorrelation below 0.05 are left alone, as in ``zyp``.

    Args:
        y: The series to test.
        x: Matching independent variable. Defaults to ``1..n``.

    Returns:
        Dictionary with ``trend`` (Theil-Sen slope of the prewhitened
        series), ``tau``, ``p_value``, ``autocorrelation`` (the lag-1 value
        that drove the prewhitening) and ``iterations``.
    """
    data = np.asarray(y, dtype=float)
    n = data.size
    t = np.arange(1, n + 1, dtype=float) if x is None else np.asarray(x, dtype=float)

    empty = {"trend": np.nan, "tau": np.nan, "p_value": np.nan,
             "autocorrelation": np.nan, "iterations": 0}
    if n < 4:
        return empty

    autocorr = _lag1_autocorrelation(data)
    if not np.isfinite(autocorr):
        return empty

    if autocorr < _AUTOCORR_THRESHOLD:
        series, series_t = data, t
    else:
        series = data[1:] - autocorr * data[:-1]
        series_t = t[:-1]

    usable = np.isfinite(series)
    if usable.sum() <= 3 or int(np.count_nonzero(series[usable])) < 3:
        return empty

    trend = theil_sen_slope(series_t[usable], series[usable])

    iterations = 1
    autocorr_0 = autocorr
    if autocorr >= _AUTOCORR_THRESHOLD:
        trend_0 = trend
        while iterations < _MAX_ITERATIONS:
            detrended = data - trend * t
            autocorr = _lag1_autocorrelation(detrended)
            if not np.isfinite(autocorr):
                break
            if autocorr < _AUTOCORR_THRESHOLD and abs(autocorr - autocorr_0) <= _AUTOCORR_TOLERANCE:
                break

            series = (data[1:] - autocorr * data[:-1]) / (1.0 - autocorr)
            usable = np.isfinite(series)
            if usable.sum() <= 3:
                break
            trend = theil_sen_slope(series_t[usable], series[usable])
            iterations += 1

            if trend != 0 and np.isfinite(trend):
                relative_change = abs((trend - trend_0) / trend)
                if relative_change <= _TREND_TOLERANCE and abs(autocorr - autocorr_0) <= _AUTOCORR_TOLERANCE:
                    break
            trend_0, autocorr_0 = trend, autocorr

    result = mann_kendall(series_t[usable], series[usable])
    return {
        "trend": trend,
        "tau": result["tau"],
        "p_value": result["p_value"],
        "autocorrelation": autocorr_0,
        "iterations": iterations,
    }

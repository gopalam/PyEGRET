"""
WRTDS-Kalman (WRTDS_K): daily estimates that honour the observed samples.

Mirrors R EGRET's ``WRTDSKalman()`` and its helpers ``cleanUp()``,
``randomSubset()``, ``populateDailySamp()``, ``genmissing()`` and
``precompute_L()`` (R/kalman_functions.R).

Ordinary WRTDS estimates each day from the surface alone, so on the day after
a sample that came in far above the model the estimate takes no notice. The
model's residuals are, however, strongly autocorrelated from day to day. WRTDS_K
treats the standardized residual as an AR(1) process with lag-one correlation
``rho``: on sampled days the residual is known, and on the days between two
samples it is simulated conditional on the residuals at both ends. Averaging
``niter`` such simulations gives ``GenFlux`` and ``GenConc``.

Reference: Zhang, Q. and Hirsch, R.M. (2019), River water-quality concentration
and flux estimation can be improved by accounting for serial correlation
through an autoregressive model, Water Resources Research 55, 9705-9723.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.data.quality import make_augmented_sample
from pyegret.elist import has_model

__all__ = [
    "wrtds_kalman",
    "random_subset",
    "clean_up",
    "populate_daily_samp",
    "precompute_l",
    "gen_missing",
]


def random_subset(df: pd.DataFrame, col_name: str = "Julian", seed=None) -> pd.DataFrame:
    """Keep one randomly chosen row for each repeated value of a column.

    Mirrors R ``randomSubset()``. Used to keep one sample per day when a day
    was sampled more than once.

    Args:
        df: DataFrame, typically the Sample.
        col_name: Column whose repeated values are thinned. Default ``Julian``.
        seed: Seed or ``numpy.random.Generator``.

    Returns:
        DataFrame with each value of ``col_name`` appearing once, sorted by it.
    """
    values = df[col_name].to_numpy()
    unique_values, counts = np.unique(values, return_counts=True)
    repeated = unique_values[counts > 1]
    if len(repeated) == 0:
        return df

    rng = np.random.default_rng(seed)
    keep = ~np.isin(values, repeated)
    for value in repeated:
        rows = np.flatnonzero(values == value)
        chosen = rng.choice(rows)
        keep[chosen] = True
    subset = df[keep]
    subset = subset.sort_values(col_name, kind="stable")
    subset = subset.reset_index(drop=True)
    return subset


def clean_up(elist: dict, seed=None) -> dict:
    """One sample per day, with censored values replaced by random draws.

    Mirrors R ``cleanUp()``: thin same-day samples with :func:`random_subset`,
    then draw a value for every censored sample from the fitted model
    (:func:`~pyegret.data.quality.make_augmented_sample`) and treat the drawn
    values as if they had been measured.

    Args:
        elist: eList dict with a cross-validated Sample (``yHat``, ``SE``).
        seed: Seed or ``numpy.random.Generator``.

    Returns:
        A new eList with the cleaned Sample.
    """
    rng = np.random.default_rng(seed)
    subset_seed = int(rng.integers(0, 2**31 - 1))
    draw_seed = int(rng.integers(0, 2**31 - 1))

    elist = dict(elist)
    elist["Sample"] = random_subset(elist["Sample"], "Julian", seed=subset_seed)
    augmented = make_augmented_sample(elist, seed=draw_seed)
    sample = augmented["Sample"].copy()
    sample["Uncen"] = 1
    sample["ConcLow"] = sample["rObserved"]
    sample["ConcHigh"] = sample["rObserved"]
    sample["ConcAve"] = sample["rObserved"]
    elist["Sample"] = sample
    return elist


def populate_daily_samp(elist: dict) -> pd.DataFrame:
    """Daily with the sampled concentration and standardized residual added.

    Mirrors R ``populateDailySamp()``. Adds ``trueConc`` (the sample's
    ConcAve, NaN on unsampled days), ``trueFlux`` and
    ``stdResid = (log(trueConc) - yHat) / SE``.

    Args:
        elist: eList dict whose Daily has ``yHat`` and ``SE`` and whose
            Sample has one row per day (see :func:`clean_up`).

    Returns:
        The augmented Daily DataFrame, in date order.
    """
    daily = elist["Daily"]
    sample = elist["Sample"]
    true_conc = pd.DataFrame({
        "Julian": sample["Julian"].to_numpy(),
        "trueConc": sample["ConcAve"].to_numpy(dtype=float),
    })
    merged = pd.merge(daily, true_conc, on="Julian", how="left")
    merged = merged.sort_values("Julian", kind="stable")
    merged = merged.reset_index(drop=True)

    conc = merged["trueConc"].to_numpy(dtype=float)
    merged["trueFlux"] = conc * merged["Q"].to_numpy(dtype=float) * 86.4
    y_hat = merged["yHat"].to_numpy(dtype=float)
    se = merged["SE"].to_numpy(dtype=float)
    merged["stdResid"] = (np.log(conc) - y_hat) / se
    return merged


def precompute_l(rho: float, n_max: int) -> dict:
    """Cholesky factor of the precision matrix of an AR(1) bridge.

    Mirrors R ``precompute_L()``. The interior points of an AR(1) segment
    pinned at both ends have a tridiagonal precision matrix with diagonal
    ``(1 + rho^2) / (1 - rho^2)`` and off-diagonal ``-rho / (1 - rho^2)``. Its
    Cholesky factor is lower bidiagonal; the entries do not depend on the
    segment length, so one factor computed for the longest segment serves
    every shorter one.

    Args:
        rho: Lag-one autocorrelation.
        n_max: Length of the longest segment, end points included.

    Returns:
        Dict with ``Ldiag`` (diagonal) and ``Lsub`` (sub-diagonal; entry 0
        unused) arrays of length ``n_max - 2``.
    """
    s2 = 1.0 - rho ** 2
    m = n_max - 2
    d = (1.0 + rho ** 2) / s2
    o = -rho / s2
    l_diag = np.zeros(m)
    l_sub = np.zeros(m)
    l_diag[0] = np.sqrt(d)
    for i in range(1, m):
        l_sub[i] = o / l_diag[i - 1]
        l_diag[i] = np.sqrt(d - l_sub[i] ** 2)
    return {"Ldiag": l_diag, "Lsub": l_sub}


def gen_missing(x1: float, xn: float, rho: float, n: int, l_full: dict,
                rng: np.random.Generator) -> np.ndarray:
    """Simulate an AR(1) segment between two known end values.

    Mirrors R ``genmissing()``. The conditional mean of the interior points
    is found by solving ``Omega mu = b`` with the bidiagonal factor, and the
    noise is added by back-substituting standard normal draws, which gives
    exactly the conditional covariance ``Omega^-1``.

    Args:
        x1: Value at the start of the segment.
        xn: Value at the end.
        rho: Lag-one autocorrelation.
        n: Segment length, both ends included.
        l_full: Output of :func:`precompute_l` for at least this length.
        rng: Random generator.

    Returns:
        Array of length ``n``: ``x1``, the simulated interior, ``xn``.
    """
    if n == 2:
        return np.array([x1, xn])

    s2 = 1.0 - rho ** 2
    m = n - 2
    l_diag = l_full["Ldiag"][:m]
    l_sub = l_full["Lsub"][:m]

    b = np.zeros(m)
    if m == 1:
        b[0] = (rho / s2) * (x1 + xn)
    else:
        b[0] = (rho / s2) * x1
        b[m - 1] = (rho / s2) * xn

    # forward pass: L w = b
    w = np.zeros(m)
    w[0] = b[0] / l_diag[0]
    for i in range(1, m):
        w[i] = (b[i] - l_sub[i] * w[i - 1]) / l_diag[i]

    # backward pass: L^T mu = w
    mu = np.zeros(m)
    mu[m - 1] = w[m - 1] / l_diag[m - 1]
    for i in range(m - 2, -1, -1):
        mu[i] = (w[i] - l_sub[i + 1] * mu[i + 1]) / l_diag[i]

    # noise: L^T y = z
    z = rng.standard_normal(m)
    y = np.zeros(m)
    y[m - 1] = z[m - 1] / l_diag[m - 1]
    for i in range(m - 2, -1, -1):
        y[i] = (z[i] - l_sub[i + 1] * y[i + 1]) / l_diag[i]

    segment = np.concatenate([[x1], mu + y, [xn]])
    return segment


def _missing_runs(x: np.ndarray) -> list:
    """(start, end) index pairs, in the zero-padded series, of each gap to fill.

    The series is padded with a known 0 at both ends, as in R, so every run
    of missing values has a known value on each side. Each pair spans the
    gap plus those two known neighbours.
    """
    padded_missing = np.concatenate([[False], np.isnan(x), [False]])
    runs = []
    i = 0
    n = len(padded_missing)
    while i < n:
        if padded_missing[i]:
            start = i
            while i < n and padded_missing[i]:
                i = i + 1
            runs.append((start - 1, i))
        else:
            i = i + 1
    return runs


def wrtds_kalman(elist: dict, rho: float = 0.90, niter: int = 200,
                 seed=None, verbose: bool = True) -> dict:
    """Add WRTDS-Kalman estimates ``GenConc`` and ``GenFlux`` to Daily.

    Mirrors R ``WRTDSKalman()``. For each of ``niter`` iterations: thin
    same-day samples and draw censored values (:func:`clean_up`), compute the
    standardized residual on every sampled day, fill every run of unsampled
    days with a conditional AR(1) simulation (:func:`gen_missing`), and turn
    the residuals back into flux. ``GenFlux`` is the mean flux across
    iterations and ``GenConc = GenFlux / (Q * 86.4)``.

    Results vary slightly between runs because of the random draws; pass
    ``seed`` to fix them. They will not reproduce R's numbers digit for
    digit, because R and NumPy use different random generators, but they
    agree in distribution.

    Args:
        elist: eList dict after :func:`~pyegret.model.model_estimation.model_estimation`
            (Daily needs ``yHat`` and ``SE``; Sample needs ``yHat`` and ``SE``).
        rho: Lag-one autocorrelation of the daily residuals. Default 0.90,
            EGRET's default; Zhang and Hirsch found 0.85-0.95 works for most
            constituents.
        niter: Number of simulations to average. Default 200.
        seed: Seed for reproducible results.
        verbose: Print progress. Default True.

    Returns:
        A new eList whose Daily has ``GenConc`` and ``GenFlux``. The input
        eList is not changed.

    Raises:
        ValueError: If the model has not been fitted, or if every day is
            sampled (nothing to fill).
    """
    if not has_model(elist):
        raise ValueError("Run model_estimation on the eList before wrtds_kalman.")
    daily = elist["Daily"]
    if "yHat" not in daily.columns or "SE" not in daily.columns:
        raise ValueError(
            "Daily has no yHat / SE columns; re-run model_estimation so "
            "est_daily_from_surfaces adds them."
        )

    rng = np.random.default_rng(seed)

    first = clean_up(elist, seed=rng)
    first_daily = populate_daily_samp(first)
    if np.sum(np.isnan(first_daily["trueConc"].to_numpy())) == 0:
        raise ValueError(
            "There are known concentration values for every day in the Daily data frame."
        )

    n_days = len(first_daily)
    runs = _missing_runs(first_daily["stdResid"].to_numpy(dtype=float))
    l_full = precompute_l(rho, n_days + 2)

    q = first_daily["Q"].to_numpy(dtype=float)
    y_hat = first_daily["yHat"].to_numpy(dtype=float)
    se = first_daily["SE"].to_numpy(dtype=float)

    if niter > 5000:
        raise ValueError("niter can be at most 5000, as in R.")
    seeds = rng.choice(np.arange(1, 5001), size=niter, replace=False)

    gen_flux_sum = np.zeros(n_days)
    gen_flux_count = np.zeros(n_days)
    next_report = 10
    for k in range(niter):
        iteration_rng = np.random.default_rng(int(seeds[k]))
        cleaned = clean_up(elist, seed=iteration_rng)
        local_daily = populate_daily_samp(cleaned)
        x = local_daily["stdResid"].to_numpy(dtype=float)
        padded = np.concatenate([[0.0], x, [0.0]])
        for start, end in runs:
            n_fill = end - start + 1
            padded[start:end + 1] = gen_missing(padded[start], padded[end], rho,
                                                n_fill, l_full, iteration_rng)
        resid = padded[1:n_days + 1]
        conc = np.exp(resid * se + y_hat)
        flux = conc * q * 86.4
        ok = ~np.isnan(flux)
        gen_flux_sum[ok] = gen_flux_sum[ok] + flux[ok]
        gen_flux_count[ok] = gen_flux_count[ok] + 1

        if verbose and 100 * (k + 1) / niter >= next_report:
            print(f"{next_report}", end="\t" if next_report % 100 else "\n", flush=True)
            next_report = next_report + 10

    with np.errstate(invalid="ignore", divide="ignore"):
        gen_flux = gen_flux_sum / gen_flux_count

    # populate_daily_samp sorted by Julian; put the results back in the
    # order of the eList's own Daily.
    julian_sorted = first_daily["Julian"].to_numpy()
    order = np.argsort(julian_sorted)
    position = np.searchsorted(julian_sorted[order], daily["Julian"].to_numpy())
    gen_flux = gen_flux[order][position]

    new_daily = daily.copy()
    new_daily["GenFlux"] = gen_flux
    new_daily["GenConc"] = gen_flux / (new_daily["Q"].to_numpy(dtype=float) * 86.4)

    elist = dict(elist)
    elist["Daily"] = new_daily
    info = elist["INFO"].copy()
    info["kalman_rho"] = rho
    info["kalman_niter"] = niter
    elist["INFO"] = info
    return elist

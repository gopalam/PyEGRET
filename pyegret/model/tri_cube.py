"""
Tricube kernel weighting for WRTDS.

Every sample observation receives a weight that decays smoothly to zero as
its distance from the target grid point grows in three dimensions: time,
log-discharge, and season.

Distance definitions follow R EGRET's ``runSurvReg()`` exactly:

============  ==========================================  ===============
dimension     distance                                    default window
============  ==========================================  ===============
time          ``|DecYear - target_year|``, in years       7.0
log-Q         ``|LogQ - target_logQ|``, in log units      2.0
season        ``min(dt mod 1, 1 - dt mod 1)``, in years   0.5
============  ==========================================  ===============

The season distance is the *shorter way round the calendar* between the
sample and the target, so it lies in ``[0, 0.5]`` and is zero when the two
fall on the same day of the year regardless of how many years apart they
are.  A window of 0.5 therefore spans a half year with smooth decay and
reaches zero only at the opposite season.

.. note::
   Earlier versions measured seasonal distance as the chord length between
   ``(sin, cos)`` of decimal year, which ranges over ``[0, 2]``.  With
   ``window_s = 0.5`` that gave zero weight to anything more than about 29
   days away in season - roughly a tenth of the intended kernel width - and
   collapsed the effective sample size at every grid point.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "tri_cube",
    "season_distance",
    "compute_weights",
    "compute_adaptive_weights",
]

# EGRET defaults (modelEstimation / runSurvReg)
DEFAULT_WINDOW_Y = 7.0
DEFAULT_WINDOW_Q = 2.0
DEFAULT_WINDOW_S = 0.5
DEFAULT_MIN_NUM_OBS = 100
DEFAULT_MIN_NUM_UNCEN = 50

# Season distance can never exceed half a year, so widening past 0.5
# would have no effect.
_MAX_WINDOW_S = 0.5
_EXPANSION_FACTOR = 1.1
_MAX_EXPANSIONS = 60


def tri_cube(x: np.ndarray, half_window: float) -> np.ndarray:
    """Tricube kernel weight function.

    Returns ``(1 - |x/h|^3)^3`` for ``|x| < h``, else ``0``.
    Fully vectorized - accepts arrays of any shape.

    Args:
        x: Distance values (any shape ndarray or scalar).
        half_window: Bandwidth ``h > 0``.

    Returns:
        ndarray of the same shape as ``x`` with values in ``[0, 1]``.
    """
    x = np.asarray(x, dtype=float)
    u = np.abs(x) / half_window
    return np.where(u < 1.0, (1.0 - u ** 3) ** 3, 0.0)


def season_distance(sample_years: np.ndarray, target_year: float) -> np.ndarray:
    """Seasonal distance in years, in ``[0, 0.5]``.

    Mirrors R EGRET::

        distTime   <- abs(DecYear - estPtYear)
        distSeason <- pmin(distTime %% 1, 1 - (distTime %% 1))

    Two dates one year apart have a seasonal distance of zero; two dates six
    months apart have the maximum distance of 0.5.

    Args:
        sample_years: 1-D array of sample decimal years.
        target_year: Decimal year of the target grid point.

    Returns:
        ndarray of seasonal distances, same shape as ``sample_years``.
    """
    frac = np.abs(np.asarray(sample_years, dtype=float) - target_year) % 1.0
    return np.minimum(frac, 1.0 - frac)


def compute_weights(
    target_year: float,
    target_logQ: float,
    sample_years: np.ndarray,
    sample_logQ: np.ndarray,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
) -> np.ndarray:
    """Combined WRTDS tricube weights for all sample observations.

    The weight is the element-wise product of three independent tricube
    kernels - time, log-discharge and season - as defined in the module
    docstring.

    Args:
        target_year: Decimal year of the target grid point.
        target_logQ: log(Q) of the target grid point.
        sample_years: 1-D array of sample decimal years (length n).
        sample_logQ: 1-D array of sample log(Q) values (length n).
        window_y: Tricube half-window for time, in years. Default 7.0.
        window_q: Tricube half-window for log-discharge. Default 2.0.
        window_s: Tricube half-window for season, in years. Default 0.5.

    Returns:
        1-D ndarray of length n with non-negative combined weights.
        A sum of zero means no data lies near this grid point.
    """
    sample_years = np.asarray(sample_years, dtype=float)
    sample_logQ = np.asarray(sample_logQ, dtype=float)

    w_time = tri_cube(sample_years - target_year, window_y)
    w_logq = tri_cube(sample_logQ - target_logQ, window_q)
    w_season = tri_cube(season_distance(sample_years, target_year), window_s)

    return w_time * w_logq * w_season


def compute_adaptive_weights(
    target_year: float,
    target_logQ: float,
    sample_years: np.ndarray,
    sample_logQ: np.ndarray,
    sample_uncen: np.ndarray,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    edge_adjust: bool = True,
    dec_low: float | None = None,
    dec_high: float | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Tricube weights with EGRET's window expansion and edge adjustment.

    Mirrors the ``repeat`` loop in R ``runSurvReg()``.  A fixed window leaves
    sparse corners of the (year, logQ) grid with too few observations to
    support a regression, which is what produces NaN grid cells and wildly
    unstable fits.  EGRET instead widens all three windows by 10 % at a time
    until enough observations carry non-zero weight.

    Edge adjustment (EGRET's ``edgeAdjust``) widens the *time* window near the
    start and end of the record, where a symmetric window would run off the
    end of the data and bias the fit toward the interior.

    Args:
        target_year: Decimal year of the target grid point.
        target_logQ: log(Q) of the target grid point.
        sample_years: 1-D array of sample decimal years (length n).
        sample_logQ: 1-D array of sample log(Q) values (length n).
        sample_uncen: 1-D array of censoring flags, 1 = detected.
        window_y: Starting half-window for time, in years. Default 7.0.
        window_q: Starting half-window for log-discharge. Default 2.0.
        window_s: Starting half-window for season, in years. Default 0.5.
        min_num_obs: Minimum observations with non-zero weight. Default 100.
            Automatically reduced to the record length when the record is
            shorter than this, so short records still converge.
        min_num_uncen: Minimum *uncensored* observations with non-zero
            weight. Default 50, reduced the same way.
        edge_adjust: Widen the time window near the ends of the record.
            Default True.
        dec_low: Earliest sample decimal year. Computed from
            ``sample_years`` when None.
        dec_high: Latest sample decimal year. Computed from
            ``sample_years`` when None.

    Returns:
        ``(weights, (window_y_used, window_q_used, window_s_used))``.
        The windows actually used are returned so callers can report how far
        the kernel had to stretch - a large stretch means the estimate at
        that grid point rests on distant data.
    """
    sample_years = np.asarray(sample_years, dtype=float)
    sample_logQ = np.asarray(sample_logQ, dtype=float)
    sample_uncen = np.asarray(sample_uncen)

    n = sample_years.size
    n_uncen = int((sample_uncen == 1).sum())

    # A record shorter than the minimum can never satisfy it; asking for more
    # than exists would expand the windows until they covered everything and
    # then give up. Cap the requirement at what the record actually holds.
    min_num_obs = min(min_num_obs, n)
    min_num_uncen = min(min_num_uncen, n_uncen)

    if dec_low is None:
        dec_low = float(sample_years.min())
    if dec_high is None:
        dec_high = float(sample_years.max())

    # -- Edge adjustment: stretch the time window near the record ends -----
    # EGRET run_WRTDS():
    #   distTime <- min(estY - DecLow, DecHigh - estY)
    #   tempWindowY <- if (distTime > tempWindowY) tempWindowY
    #                  else (2 * tempWindowY - distTime)
    w_y = float(window_y)
    if edge_adjust:
        dist_time = min(target_year - dec_low, dec_high - target_year)
        if dist_time <= w_y:
            w_y = 2.0 * w_y - dist_time

    w_q = float(window_q)
    w_s = float(window_s)

    # -- Expansion loop ----------------------------------------------------
    weights = compute_weights(
        target_year, target_logQ, sample_years, sample_logQ, w_y, w_q, w_s
    )
    for _ in range(_MAX_EXPANSIONS):
        positive = weights > 0
        if (positive.sum() >= min_num_obs
                and int((sample_uncen[positive] == 1).sum()) >= min_num_uncen):
            break
        w_y *= _EXPANSION_FACTOR
        w_q *= _EXPANSION_FACTOR
        w_s = min(w_s * _EXPANSION_FACTOR, _MAX_WINDOW_S)
        weights = compute_weights(
            target_year, target_logQ, sample_years, sample_logQ, w_y, w_q, w_s
        )

    return weights, (w_y, w_q, w_s)

"""
Top-level WRTDS model estimation pipeline.
Mirrors R ``modelEstimation()``.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyegret.elist import has_sample
from pyegret.model.est_cross_val import est_cross_val
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import (
    N_VECTOR_LOGQ,
    POINTS_PER_YEAR,
    est_surfaces,
)
from pyegret.model.tri_cube import (
    DEFAULT_MIN_NUM_OBS,
    DEFAULT_MIN_NUM_UNCEN,
    DEFAULT_WINDOW_Q,
    DEFAULT_WINDOW_S,
    DEFAULT_WINDOW_Y,
)

__all__ = ["model_estimation"]


def model_estimation(
    elist: dict,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    edge_adjust: bool = True,
    points_per_year: int = POINTS_PER_YEAR,
    n_vector_logQ: int = N_VECTOR_LOGQ,
    n_vector_year: int | None = None,
    cross_validate: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the complete WRTDS model estimation pipeline.

    Mirrors R ``modelEstimation()``:

    1. Cross-validate: leave-one-out predictions for every sample, which is
       what honest residual diagnostics are defined on.
    2. Estimate the concentration surface over the (year, logQ) grid.
    3. Interpolate it onto every daily row, producing ConcDay, FluxDay,
       FNConc and FNFlux.

    Args:
        elist: eList dict with Daily and Sample DataFrames.
        window_y: Starting tricube half-window for time, in years. Default 7.0.
        window_q: Starting half-window for log-discharge. Default 2.0.
        window_s: Starting half-window for season, in years. Default 0.5.
        min_num_obs: Minimum weighted observations per grid point. Default 100.
        min_num_uncen: Minimum uncensored weighted observations. Default 50.
        edge_adjust: Widen the time window near the record ends. Default True.
        points_per_year: Year-axis grid resolution. Default 16.
        n_vector_logQ: Number of log-discharge grid points. Default 14.
        n_vector_year: Override the year-axis point count. For fast tests.
        cross_validate: Fill Sample ``yHat`` / ``SE`` / ``ConcHat`` by
            leave-one-out cross-validation, as EGRET does. Default True.
            Setting False substitutes in-sample predictions read off the
            fitted surface, which is faster but makes every residual
            statistic optimistic.
        verbose: Print progress. Default True.

    Returns:
        A new eList (the input is not changed), with ``surfaces`` fitted,
        ``Daily`` augmented with
        ``ConcDay`` / ``FluxDay`` / ``FNConc`` / ``FNFlux``, ``Sample``
        augmented with ``yHat`` / ``SE`` / ``ConcHat`` / ``resid``, and the
        settings recorded in ``INFO`` (``windowY``, ``windowQ``, ``windowS``,
        ``minNumObs``, ``minNumUncen``, ``edgeAdjust``), as R does.

    Raises:
        ValueError: If ``elist`` has no Sample data.
    """
    if not has_sample(elist):
        raise ValueError(
            "eList has no Sample data. Load samples before calling model_estimation()."
        )
    elist = dict(elist)   # work on a copy; the caller's eList is not changed

    if cross_validate:
        elist["Sample"] = est_cross_val(
            elist,
            window_y=window_y,
            window_q=window_q,
            window_s=window_s,
            min_num_obs=min_num_obs,
            min_num_uncen=min_num_uncen,
            edge_adjust=edge_adjust,
            verbose=verbose,
        )

    elist["surfaces"] = est_surfaces(
        elist,
        window_y=window_y,
        window_q=window_q,
        window_s=window_s,
        min_num_obs=min_num_obs,
        min_num_uncen=min_num_uncen,
        edge_adjust=edge_adjust,
        points_per_year=points_per_year,
        n_vector_logQ=n_vector_logQ,
        n_vector_year=n_vector_year,
        verbose=verbose,
    )

    elist["Daily"] = est_daily_from_surfaces(elist)

    if not cross_validate:
        elist = _est_sample_predictions(elist)

    # As R's setUpEstimation does, record the settings in INFO, where
    # EGRETci's functions read them.
    info = elist["INFO"].copy()
    info["windowY"] = window_y
    info["windowQ"] = window_q
    info["windowS"] = window_s
    info["minNumObs"] = min_num_obs
    info["minNumUncen"] = min_num_uncen
    info["edgeAdjust"] = edge_adjust
    elist["INFO"] = info

    if verbose:
        print("WRTDS model estimation complete.")
    return elist


def _est_sample_predictions(elist: dict) -> dict:
    """Predict at each Sample observation from the fitted surface.

    Adds or updates Sample columns ``yHat``, ``SE``, ``ConcHat`` and
    ``resid``.

    .. note::
       These are *in-sample* predictions: each observation helped fit the
       surface it is being scored against, so residual statistics computed
       from them are optimistic. This is the fallback used only when
       ``model_estimation(..., cross_validate=False)``; the default path
       fills these columns from
       :func:`~pyegret.model.est_cross_val.est_cross_val` instead, matching
       EGRET.

    Args:
        elist: eList dict with fitted surfaces and a Sample DataFrame.

    Returns:
        The eList with Sample updated.
    """
    elist = dict(elist)
    sample = elist["Sample"].copy()
    surfaces = elist["surfaces"]
    v_year = np.asarray(elist["vector_year"], dtype=float)
    v_logQ = np.asarray(elist["vector_logQ"], dtype=float)

    kw = dict(method="linear", bounds_error=False, fill_value=None)
    interp_yhat = RegularGridInterpolator((v_year, v_logQ), surfaces[:, :, 0], **kw)
    interp_se = RegularGridInterpolator((v_year, v_logQ), surfaces[:, :, 1], **kw)

    pts = np.column_stack([
        np.clip(sample["DecYear"].to_numpy(dtype=float), v_year[0], v_year[-1]),
        np.clip(sample["LogQ"].to_numpy(dtype=float), v_logQ[0], v_logQ[-1]),
    ])

    yhat = np.asarray(interp_yhat(pts), dtype=float)
    se = np.asarray(interp_se(pts), dtype=float)

    sample["yHat"] = yhat
    sample["SE"] = se
    sample["ConcHat"] = np.exp(yhat + 0.5 * se ** 2)
    with np.errstate(divide="ignore"):
        log_conc_ave = np.log(sample["ConcAve"].to_numpy(dtype=float))
    sample["resid"] = log_conc_ave - yhat

    elist["Sample"] = sample
    return elist

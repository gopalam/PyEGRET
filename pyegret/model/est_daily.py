"""
Interpolate the fitted WRTDS surface onto every day in the Daily record.

Produces ``yHat``, ``SE``, ``ConcDay``, ``FluxDay``, ``FNConc`` and ``FNFlux``,
mirroring R EGRET's ``estDailyFromSurfaces()``, ``getSurfaceEstimates()`` and
``getConcFluxFromSurface()``. ``yHat`` and ``SE`` are what WRTDS-Kalman needs.

Two details of EGRET's flow normalization are easy to get wrong and both
change the answer:

**Reference flows are grouped by day of the year.**  ``FNConc`` for a given
day is the concentration the model predicts for that date, averaged over the
discharges actually observed *on that same day of the year across all years*
(EGRET's ``allLogQsByDayOfYear``).  Averaging over every discharge in the
record instead mixes January flows into July estimates and destroys exactly
the seasonal structure flow normalization exists to preserve.

**Flux is averaged, not multiplied.**  ``FNFlux`` is the mean of
``conc x Q`` over those reference flows - not ``FNConc x Q`` for the day in
question.  The two differ whenever concentration and discharge covary, which
is almost always.

Concentration is read from the ``ConcHat`` slice of the surface
(``exp(yHat + SE^2/2)``), not from ``exp(yHat)``.  Dropping that log-normal
bias correction biases every concentration and flux low by ``exp(SE^2/2)``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


__all__ = ["est_daily_from_surfaces"]

# mg/L x m^3/s -> kg/day  (86 400 s/day / 1 000 g/kg)
FLUX_FACTOR = 86.4

# Slice index of the bias-corrected concentration in the surfaces array.
_CONC_HAT = 2


def _day_of_year(frame: pd.DataFrame) -> np.ndarray:
    """Day-of-year for each row, from the ``Day`` column or from ``Date``."""
    if "Day" in frame.columns:
        return frame["Day"].to_numpy(dtype=int)
    return frame["Date"].dt.day_of_year.to_numpy(dtype=int)


def est_daily_from_surfaces(
    elist: dict,
    flow_norm_daily: pd.DataFrame | None = None,
    surfaces: np.ndarray | None = None,
    vector_year: np.ndarray | None = None,
    vector_logQ: np.ndarray | None = None,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Interpolate the WRTDS surface onto every row of Daily.

    Mirrors R ``estDailyFromSurfaces(eList, localsurfaces, localDaily)``.
    Days that fall outside the surface's year or log-discharge range get
    NaN, as R's ``fields::interp.surface`` gives NA there; that is what lets
    ``runPairs`` fit a surface for one year only.

    Args:
        elist: eList dict with fitted ``surfaces``, ``vector_year`` and
            ``vector_logQ`` (all set by ``est_surfaces``).
        flow_norm_daily: Optional subset of Daily supplying the reference
            discharges for flow normalization. Defaults to the Daily being
            estimated, which is EGRET's standard flow normalization; passing a
            subset gives the generalized form used by ``flexFN``.
        surfaces: Use this surface instead of the eList's (R ``localsurfaces``).
            Give ``vector_year`` and ``vector_logQ`` with it.
        vector_year: Year axis of ``surfaces``.
        vector_logQ: Log-discharge axis of ``surfaces``.
        daily: Estimate for this Daily instead of the eList's (R
            ``localDaily``).

    Returns:
        A copy of Daily with six columns added:

        - ``yHat``    - fitted log-concentration
        - ``SE``      - standard error of the log-concentration
        - ``ConcDay`` - estimated concentration (mg/L)
        - ``FluxDay`` - estimated flux (kg/day)
        - ``FNConc``  - flow-normalized concentration (mg/L)
        - ``FNFlux``  - flow-normalized flux (kg/day)

    Raises:
        ValueError: If the surface or its grid axes are missing.
    """
    if surfaces is None:
        if elist.get("surfaces") is None:
            raise ValueError("eList surfaces is None - run est_surfaces first.")
        if elist.get("vector_year") is None or elist.get("vector_logQ") is None:
            raise ValueError(
                "elist is missing vector_year / vector_logQ - run est_surfaces first."
            )
        surfaces = elist["surfaces"]
        vector_year = elist["vector_year"]
        vector_logQ = elist["vector_logQ"]
    if daily is None:
        daily = elist["Daily"]
    daily = daily.copy()
    v_year = np.asarray(vector_year, dtype=float)
    v_logQ = np.asarray(vector_logQ, dtype=float)

    conc_surface = surfaces[:, :, _CONC_HAT]
    # Whole year-rows of NaN are a surface fitted for part of the record
    # (run_series, a wall); only gaps inside the fitted rows are a problem.
    fitted_rows = np.any(np.isfinite(conc_surface), axis=1)
    n_nan = int(np.sum(np.isnan(conc_surface[fitted_rows])))
    if n_nan:
        warnings.warn(
            f"{n_nan} of {conc_surface.size} surface grid points are NaN; "
            "days that interpolate through them will be NaN. Consider "
            "lowering min_num_obs or widening the starting windows.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Outside the grid the answer is NaN, as in R.
    interp = RegularGridInterpolator(
        (v_year, v_logQ), conc_surface,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    yr_q = daily["DecYear"].to_numpy(dtype=float)
    lq_q = daily["LogQ"].to_numpy(dtype=float)
    q_vals = daily["Q"].to_numpy(dtype=float)

    # -- yHat / SE / ConcDay / FluxDay (R getSurfaceEstimates) -------------
    points = np.column_stack([yr_q, lq_q])
    for name, index in (("yHat", 0), ("SE", 1)):
        slice_interp = RegularGridInterpolator(
            (v_year, v_logQ), surfaces[:, :, index],
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        daily[name] = slice_interp(points)
    conc_day = interp(points)
    daily["ConcDay"] = conc_day
    daily["FluxDay"] = conc_day * q_vals * FLUX_FACTOR

    # -- Reference discharges for flow normalization, by day of year -------
    ref = daily
    if flow_norm_daily is not None:
        ref = flow_norm_daily
    ref_doy = _day_of_year(ref)
    ref_logq_all = ref["LogQ"].to_numpy(dtype=float)
    ref_q_all = ref["Q"].to_numpy(dtype=float)
    usable = np.isfinite(ref_logq_all)
    ref_doy = ref_doy[usable]
    ref_logq_all = ref_logq_all[usable]
    ref_q_all = ref_q_all[usable]

    by_doy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in np.unique(ref_doy):
        mask = ref_doy == k
        by_doy[int(k)] = (ref_logq_all[mask], ref_q_all[mask])

    # R bin_Qs ("Bob's convention"): February 28 (day 59) and February 29
    # (day 60) share one pool of reference flows, so the leap day is not
    # normalized against only the few leap years in the record.
    if 59 in by_doy and 60 in by_doy:
        pooled_lq = np.concatenate([by_doy[59][0], by_doy[60][0]])
        pooled_q = np.concatenate([by_doy[59][1], by_doy[60][1]])
        by_doy[59] = (pooled_lq, pooled_q)
        by_doy[60] = (pooled_lq, pooled_q)

    # -- FNConc / FNFlux ---------------------------------------------------
    daily_doy = _day_of_year(daily)
    fn_conc = np.full(len(daily), np.nan)
    fn_flux = np.full(len(daily), np.nan)

    for k in np.unique(daily_doy):
        idx = np.flatnonzero(daily_doy == k)
        ref_lq, ref_q = by_doy.get(int(k), (ref_logq_all, ref_q_all))
        if ref_lq.size == 0:
            continue

        # Query every (day in this group) x (reference flow for this day of
        # the year) pair, then average across the reference flows.
        m, r = idx.size, ref_lq.size
        pts = np.column_stack([
            np.repeat(yr_q[idx], r),
            np.tile(ref_lq, m),
        ])
        conc = interp(pts)
        conc = np.reshape(conc, (m, r))

        with warnings.catch_warnings():
            # An all-NaN row is reported below as a NaN result, not here.
            warnings.simplefilter("ignore", RuntimeWarning)
            fn_conc[idx] = np.nanmean(conc, axis=1)
            fn_flux[idx] = np.nanmean(conc * ref_q[None, :], axis=1) * FLUX_FACTOR

    daily["FNConc"] = fn_conc
    daily["FNFlux"] = fn_flux

    return daily

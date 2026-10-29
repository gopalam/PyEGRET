"""
Prediction intervals for WRTDS-Kalman daily, monthly and annual estimates.

Mirrors R EGRETci 2.0.5 ``genDailyBoot()``, ``makeDailyPI()``,
``makeMonthPI()``, ``makeAnnualPI()`` and ``monthSeqToDec()``.

``gen_daily_boot`` builds many plausible daily flux records: for each of
``n_boot`` block-bootstrap resamples it refits the WRTDS model (the
estimation error), and from each refit it draws ``n_kalman`` WRTDS-Kalman
records (the day-to-day variability). The percentiles of those records, by
day, month or year, are the prediction intervals.

**A deliberate difference from EGRETci 2.0.5.** In R's ``genDailyBoot`` the
surface fitted to each resample is attached to the eList, but
``WRTDSKalman`` takes ``yHat`` and ``SE`` from ``Daily``, which still holds the
original model's values; so the resampling has no effect, and two bootstrap
replicates with the same Kalman seed are identical (checked in R 4.3.3 with
EGRET 3.0.12 and EGRETci 2.0.5). EGRETci's documentation says each replicate
"is used to estimate the WRTDS model (we can think of this as the estimation
error)", and PyEGRET does that: ``Daily`` and ``Sample`` predictions are
recomputed from each replicate's surface before the Kalman step. Pass
``boot_surfaces=False`` to reproduce EGRETci 2.0.5's behaviour instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.setup_years import setup_years
from pyegret.ci.bootstrap import block_sample_with_state, random_seed
from pyegret.ci.r_random import r_set_seed
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import est_surfaces
from pyegret.model.kalman import wrtds_kalman
from pyegret.model.model_estimation import _est_sample_predictions
from pyegret.utils.units import get_flux_unit

__all__ = [
    "gen_daily_boot",
    "make_daily_pi",
    "make_month_pi",
    "make_annual_pi",
    "month_seq_to_dec",
    "PI_PROBS",
]

# The percentiles EGRETci reports, and their column names.
PI_PROBS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.975, 0.99]
_PI_NAMES = ["p1", "p2.5", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p97.5", "p99"]


def _info_number(info: pd.DataFrame, name: str):
    if name not in info.columns:
        raise ValueError("Run model_estimation on the eList before gen_daily_boot "
                         f"(INFO has no {name}).")
    return info[name].iloc[0]


def gen_daily_boot(elist: dict, n_boot: int = 10, n_kalman: int = 10, rho: float = 0.9,
                   set_seed: int | None = None, boot_surfaces: bool = True,
                   verbose: bool = True) -> np.ndarray:
    """Many plausible daily flux records, for prediction intervals. Mirrors R ``genDailyBoot()``.

    Args:
        elist: eList dict after ``model_estimation`` (INFO must record the
            windows and minimums) and usually ``wrtds_kalman``.
        n_boot: Number of bootstrap refits of the model.
        n_kalman: Number of WRTDS-Kalman records drawn from each refit.
        rho: Lag-one autocorrelation of the daily residuals. Default 0.9.
        set_seed: Seed. The resamples use R's generator seeded with it, and
            the Kalman draws use ``set_seed + i`` (``i`` = 1 ... n_kalman),
            as in R. None gives different results each run.
        boot_surfaces: Use each refit's surface in the Kalman step (default;
            see the module notes). False reproduces EGRETci 2.0.5, in which
            the refits have no effect.
        verbose: Print progress.

    Returns:
        Array of daily flux (kg/day), one row per day of Daily and
        ``n_boot * n_kalman`` columns, ordered by refit then Kalman draw.
    """
    info = elist["INFO"]
    settings = {
        "window_y": float(_info_number(info, "windowY")),
        "window_q": float(_info_number(info, "windowQ")),
        "window_s": float(_info_number(info, "windowS")),
        "min_num_obs": _info_number(info, "minNumObs"),
        "min_num_uncen": _info_number(info, "minNumUncen"),
    }
    edge_adjust = True
    if "edgeAdjust" in info.columns:
        edge_adjust = bool(info["edgeAdjust"].iloc[0])

    daily = elist["Daily"]
    sample = elist["Sample"]
    n_days = len(daily)
    out = np.full((n_days, n_boot * n_kalman), np.nan)
    if set_seed is None:
        state = r_set_seed(random_seed())
    else:
        state = r_set_seed(set_seed)

    for i_boot in range(1, n_boot + 1):
        if verbose:
            print(f"Boot: {i_boot}")
        resampled, state = block_sample_with_state(sample, 200, state)
        refit = dict(elist)
        if boot_surfaces:
            boot = {"INFO": info, "Daily": daily, "Sample": resampled,
                    "surfaces": None, "vector_year": None, "vector_logQ": None}
            surfaces = est_surfaces(boot, edge_adjust=edge_adjust, verbose=False, **settings)
            refit["surfaces"] = surfaces
            refit["vector_year"] = boot["vector_year"]
            refit["vector_logQ"] = boot["vector_logQ"]
            refit["Daily"] = est_daily_from_surfaces(refit)
            refit = _est_sample_predictions(refit)
        # Otherwise, as in EGRETci 2.0.5, the refit would have no effect, so it
        # is not fitted at all; the resample is still drawn, keeping the
        # random-number stream as R's.

        for i_kalman in range(1, n_kalman + 1):
            seed = None
            if set_seed is not None:
                seed = set_seed + i_kalman
            kalman = wrtds_kalman(refit, rho=rho, niter=1, seed=seed, verbose=False)
            column = (i_boot - 1) * n_kalman + (i_kalman - 1)
            out[:, column] = kalman["Daily"]["GenFlux"].to_numpy(dtype=float)
    return out


def _percentile_table(values: np.ndarray) -> np.ndarray:
    """EGRETci's percentiles (R type 6, missing values ignored) along each row."""
    return np.nanquantile(values, PI_PROBS, axis=1, method="weibull").T


def _named_frame(lead: dict, table: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(lead)
    for j, name in enumerate(_PI_NAMES):
        frame[name] = table[:, j]
    return frame


def make_daily_pi(daily_boot_out: np.ndarray, elist: dict, flux_unit=3) -> dict:
    """Daily prediction intervals. Mirrors R ``makeDailyPI()``.

    Args:
        daily_boot_out: Result of :func:`gen_daily_boot`.
        elist: The eList it was made from.
        flux_unit: Kept for parity with R, which does not apply it here:
            daily flux percentiles are in kg/day.

    Returns:
        Dict with ``flux`` (kg/day) and ``conc`` (mg/L) DataFrames: ``Date``,
        ``DecYear`` and the percentiles ``p1`` ... ``p99``.
    """
    daily = elist["Daily"]
    q = daily["Q"].to_numpy(dtype=float)
    flux_table = _percentile_table(daily_boot_out)
    conc_table = _percentile_table(daily_boot_out / (86.4 * q[:, None]))
    lead = {"Date": daily["Date"].to_numpy(), "DecYear": daily["DecYear"].to_numpy()}
    return {"flux": _named_frame(lead, flux_table), "conc": _named_frame(lead, conc_table)}


def make_month_pi(daily_boot_out: np.ndarray, elist: dict, flux_unit=3) -> dict:
    """Monthly prediction intervals. Mirrors R ``makeMonthPI()``.

    Each record is averaged by calendar month; monthly concentration is the
    monthly mean flux over 86.4 times the monthly mean discharge.

    Returns:
        Dict with ``flux`` (in ``flux_unit``) and ``conc`` (mg/L) DataFrames:
        ``monthSeq``, ``DecYear`` and the percentiles.
    """
    daily = elist["Daily"]
    factor = get_flux_unit(flux_unit)["factor"]
    month_seq = daily["MonthSeq"].to_numpy()
    months = np.unique(month_seq)
    n_iter = daily_boot_out.shape[1]
    month_flux = np.full((len(months), n_iter), np.nan)
    ave_flow = np.full(len(months), np.nan)
    dec_year = np.full(len(months), np.nan)
    q = daily["Q"].to_numpy(dtype=float)
    years = daily["DecYear"].to_numpy(dtype=float)
    for i, month in enumerate(months):
        rows = month_seq == month
        month_flux[i] = np.mean(daily_boot_out[rows], axis=0)
        ave_flow[i] = np.mean(q[rows])
        dec_year[i] = np.mean(years[rows])
    # Concentration from kg/day, so it does not depend on flux_unit.
    month_conc = month_flux / (86.4 * ave_flow[:, None])
    lead = {"monthSeq": months, "DecYear": dec_year}
    return {"flux": _named_frame(lead, _percentile_table(month_flux * factor)),
            "conc": _named_frame(lead, _percentile_table(month_conc))}


def make_annual_pi(daily_boot_out: np.ndarray, elist: dict, pa_long: int = 12,
                   pa_start: int = 10, flux_unit=3) -> dict:
    """Annual prediction intervals. Mirrors R ``makeAnnualPI()``.

    Each record is summarized by period of analysis with ``setup_years``
    (complete years only).

    Returns:
        Dict with ``flux`` (in ``flux_unit``) and ``conc`` (mg/L) DataFrames:
        ``DecYear`` and the percentiles.
    """
    daily = elist["Daily"]
    factor = get_flux_unit(flux_unit)["factor"]
    q = daily["Q"].to_numpy(dtype=float)
    fluxes = []
    concs = []
    dec_year = None
    for i in range(daily_boot_out.shape[1]):
        local = daily.copy()
        local["GenFlux"] = daily_boot_out[:, i]
        local["GenConc"] = daily_boot_out[:, i] / (q * 86.4)
        annual = setup_years(local, pa_long=pa_long, pa_start=pa_start)
        dec_year = annual["DecYear"].to_numpy(dtype=float)
        fluxes.append(annual["GenFlux"].to_numpy(dtype=float))
        concs.append(annual["GenConc"].to_numpy(dtype=float))
    flux = np.column_stack(fluxes)
    conc = np.column_stack(concs)
    lead = {"DecYear": dec_year}
    return {"flux": _named_frame(lead, _percentile_table(flux) * factor),
            "conc": _named_frame(lead, _percentile_table(conc))}


def month_seq_to_dec(month_seq):
    """Decimal year of the middle of a MonthSeq month. Mirrors R ``monthSeqToDec()``."""
    month_seq = np.asarray(month_seq, dtype=float)
    return np.floor((month_seq - 1) / 12) + 1850 + (1 / 24) + ((month_seq - 1) % 12) / 12

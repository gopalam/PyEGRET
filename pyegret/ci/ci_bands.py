"""
Confidence bands on the annual flow-normalized series.

Mirrors R EGRETci 2.0.5 ``ciCalculations()``, ``bootAnnual()`` and
``ciBands()``: the model (or ``run_series``) is refitted to many
block-bootstrap resamples, and the spread of each year's flow-normalized
concentration and flux, reflected about the original estimate in log space,
gives the band. Plot the result with
:func:`~pyegret.ci.plots.plot_conc_hist_boot` and
:func:`~pyegret.ci.plots.plot_flux_hist_boot`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.run_series import run_series
from pyegret.analysis.setup_years import get_pa, setup_years
from pyegret.ci.bootstrap import DEFAULT_START_SEED, block_sample
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import est_surfaces
from pyegret.model.model_estimation import model_estimation

__all__ = ["boot_annual", "ci_bands", "ci_calculations"]


def _info_value(info: pd.DataFrame, name: str, default=None):
    """One INFO entry, or ``default`` if absent or missing."""
    if name not in info.columns:
        return default
    value = info[name].iloc[0]
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _integer_years(annual: pd.DataFrame) -> np.ndarray:
    """R's as.integer(DecYear): each period's decimal year, truncated."""
    dec_year = annual["DecYear"].to_numpy(dtype=float)
    years = np.trunc(dec_year)
    return years.astype(int)


def _is_series(elist: dict) -> bool:
    """True if the eList came from run_series (R's runSeries attribute)."""
    return bool(elist.get("runSeries", False))


def _series_settings(info: pd.DataFrame) -> dict:
    """The run_series arguments recorded in INFO."""
    return {
        "window_side": _info_value(info, "windowSide", 0),
        "surface_start": _info_value(info, "surfaceStart"),
        "surface_end": _info_value(info, "surfaceEnd"),
        "flow_break": bool(_info_value(info, "flowBreak", False)),
        "q1_end_date": _info_value(info, "Q1EndDate"),
        "q_start_date": _info_value(info, "QStartDate"),
        "q_end_date": _info_value(info, "QEndDate"),
        "wall": bool(_info_value(info, "wall", False)),
        "sample1_end_date": _info_value(info, "sample1EndDate"),
        "sample_start_date": _info_value(info, "sampleStartDate"),
        "sample_end_date": _info_value(info, "sampleEndDate"),
        "pa_start": int(_info_value(info, "paStart", 10)),
        "pa_long": int(_info_value(info, "paLong", 12)),
        "min_num_obs": _info_value(info, "minNumObs", 100),
        "min_num_uncen": _info_value(info, "minNumUncen", 50),
        "window_y": _info_value(info, "windowY", 7),
        "window_q": _info_value(info, "windowQ", 2),
        "window_s": _info_value(info, "windowS", 0.5),
        "edge_adjust": bool(_info_value(info, "edgeAdjust", True)),
    }


def boot_annual(elist: dict, block_length: int = 200, start_seed: int = DEFAULT_START_SEED,
                verbose: bool = False) -> pd.DataFrame:
    """Annual flow-normalized results from one bootstrap resample. Mirrors R ``bootAnnual()``.

    The Sample is resampled in blocks (seed ``start_seed``, R's generator),
    and the model refitted with the settings recorded in INFO: the same
    ``run_series`` call if the eList came from ``run_series``, otherwise
    ``est_surfaces`` as ``model_estimation`` ran it.

    Returns:
        DataFrame with ``year`` (the integer part of each period's decimal
        year), ``FNConc`` and ``FNFlux`` (kg/day).
    """
    info = elist["INFO"]
    pa_start, pa_long = get_pa(elist)
    resampled = block_sample(elist["Sample"], block_length, start_seed)
    boot = {"INFO": info, "Daily": elist["Daily"], "Sample": resampled,
            "surfaces": None, "vector_year": None, "vector_logQ": None}

    if _is_series(elist):
        settings = _series_settings(info)
        series = run_series(boot, old_surface=False, verbose=verbose, **settings)
        daily = series["Daily"]
    else:
        # R: edgeAdjust is FALSE here when INFO does not record it.
        surfaces = est_surfaces(
            boot,
            window_y=float(_info_value(info, "windowY", 7)),
            window_q=float(_info_value(info, "windowQ", 2)),
            window_s=float(_info_value(info, "windowS", 0.5)),
            min_num_obs=_info_value(info, "minNumObs", 100),
            min_num_uncen=_info_value(info, "minNumUncen", 50),
            edge_adjust=bool(_info_value(info, "edgeAdjust", False)),
            verbose=verbose,
        )
        boot["surfaces"] = surfaces
        daily = est_daily_from_surfaces(boot)

    annual = setup_years(daily, pa_long=pa_long, pa_start=pa_start)
    result = pd.DataFrame({
        "year": _integer_years(annual),
        "FNConc": annual["FNConc"].to_numpy(dtype=float),
        "FNFlux": annual["FNFlux"].to_numpy(dtype=float),
    })
    result.attrs["blockLength"] = block_length
    return result


def ci_bands(elist: dict, rep_annual_results: list, probs=(0.05, 0.95)) -> pd.DataFrame:
    """Confidence band from bootstrap annual results. Mirrors R ``ciBands()``.

    For each year, ``2 log(FN) - log(FN_boot)`` over the replicates gives the
    reflected bootstrap distribution; its ``probs`` quantiles (R's type 6),
    taken back out of logs, are the band.

    Args:
        elist: The fitted eList whose flow-normalized series the band is for.
        rep_annual_results: List of :func:`boot_annual` results.
        probs: Lower and upper probabilities. Default 5 % and 95 %.

    Returns:
        DataFrame with ``Year`` (decimal), ``FNConcLow``, ``FNConcHigh``,
        ``FNFluxLow``, ``FNFluxHigh`` (kg/day). ``attrs`` holds ``nBoot``,
        ``probs`` and ``blockLength``.
    """
    if len(probs) != 2:
        raise ValueError("Please provide only lower and upper limit in the probs argument")
    pa_start, pa_long = get_pa(elist)
    annual = setup_years(elist["Daily"], pa_long=pa_long, pa_start=pa_start)
    base = pd.DataFrame({
        "year": _integer_years(annual),
        "DecYear": annual["DecYear"].to_numpy(dtype=float),
        "FNFlux_1": annual["FNFlux"].to_numpy(dtype=float),
        "FNConc_1": annual["FNConc"].to_numpy(dtype=float),
    })
    n_boot = len(rep_annual_results)
    n_years = len(base)
    conc = np.full((n_years, n_boot), np.nan)
    flux = np.full((n_years, n_boot), np.nan)
    for i, replicate in enumerate(rep_annual_results):
        merged = base.merge(replicate, on="year", how="left")
        conc[:, i] = 2 * np.log(merged["FNConc_1"].to_numpy()) - np.log(merged["FNConc"].to_numpy())
        flux[:, i] = 2 * np.log(merged["FNFlux_1"].to_numpy()) - np.log(merged["FNFlux"].to_numpy())

    low_conc = np.full(n_years, np.nan)
    high_conc = np.full(n_years, np.nan)
    low_flux = np.full(n_years, np.nan)
    high_flux = np.full(n_years, np.nan)
    for i in range(n_years):
        good_conc = conc[i][~np.isnan(conc[i])]
        good_flux = flux[i][~np.isnan(flux[i])]
        if len(good_conc):
            quant = np.quantile(good_conc, probs, method="weibull")    # R type 6
            low_conc[i] = np.exp(quant[0])
            high_conc[i] = np.exp(quant[1])
        if len(good_flux):
            quant = np.quantile(good_flux, probs, method="weibull")
            low_flux[i] = np.exp(quant[0])
            high_flux[i] = np.exp(quant[1])

    result = pd.DataFrame({
        "Year": base["DecYear"].to_numpy(),
        "FNConcLow": low_conc, "FNConcHigh": high_conc,
        "FNFluxLow": low_flux, "FNFluxHigh": high_flux,
    })
    block_length = None
    if n_boot:
        block_length = rep_annual_results[0].attrs.get("blockLength")
    result.attrs["nBoot"] = n_boot
    result.attrs["probs"] = tuple(probs)
    result.attrs["blockLength"] = block_length
    return result


def ci_calculations(elist: dict, start_seed: int = DEFAULT_START_SEED, verbose: bool = True,
                    n_boot: int = 100, block_length: int = 200,
                    width_ci: float = 90) -> pd.DataFrame:
    """Confidence bands on the annual flow-normalized series. Mirrors R ``ciCalculations()``.

    As in R, the baseline is refitted first: ``run_series`` with the recorded
    settings if the eList came from ``run_series``, otherwise
    ``model_estimation`` with the windows and minimums in INFO. Then
    ``n_boot`` resamples (seeds ``start_seed + 1`` ... ``start_seed + n_boot``)
    are fitted with :func:`boot_annual`.

    This takes ``n_boot`` full model fits: about half a minute each for a
    record like the Choptank's, so most of an hour at the default 100.

    Args:
        elist: eList dict, after ``model_estimation`` or ``run_series``.
        start_seed: Base seed. Default 494817, EGRETci's.
        verbose: Print progress.
        n_boot: Number of bootstrap replicates.
        block_length: Block length in days.
        width_ci: Width of the band in percent. Default 90 (5th to 95th
            percentiles).

    Returns:
        The :func:`ci_bands` table.
    """
    info = elist["INFO"]
    probs = ((50 - width_ci / 2) / 100, (50 + width_ci / 2) / 100)
    if _is_series(elist):
        if verbose:
            print("\nRunning run_series to have that as a baseline for the Confidence Bands")
        settings = _series_settings(info)
        elist = run_series(elist, old_surface=True, verbose=verbose, **settings)
    else:
        if verbose:
            print("\nRunning model_estimation first to have that as a baseline for the "
                  "Confidence Bands")
        elist = model_estimation(
            elist,
            window_y=float(_info_value(info, "windowY", 7)),
            window_q=float(_info_value(info, "windowQ", 2)),
            window_s=float(_info_value(info, "windowS", 0.5)),
            min_num_obs=_info_value(info, "minNumObs", 100),
            min_num_uncen=_info_value(info, "minNumUncen", 50),
            verbose=verbose,
        )

    replicates = []
    for n in range(1, n_boot + 1):
        if verbose:
            print(n)
        replicates.append(boot_annual(elist, block_length, start_seed + n, verbose=False))
    return ci_bands(elist, replicates, probs)

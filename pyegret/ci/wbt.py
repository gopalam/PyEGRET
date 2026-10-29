"""
The WRTDS Bootstrap Test (wBT): EGRETci's original trend test.

Mirrors R EGRETci 2.0.5 ``wBT()``, ``estSliceSurfacesSimpleAlt()``,
``paVector()``, ``makeCombo()`` and ``makeTwoYearsResults()``.

``wBT`` compares two years with stationary flow normalization and stops early
once the direction of the trend is clear. EGRETci now recommends
:func:`~pyegret.analysis.run_pairs` with
:func:`~pyegret.ci.boot_trends.run_pairs_boot` instead, which also handles
generalized flow normalization; ``wbt`` is here for comparison with earlier
studies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.setup_years import get_pa, setup_years
from pyegret.ci.boot_trends import KG_DAY_TO_MILLION_KG_YEAR
from pyegret.ci.bootstrap import (DEFAULT_START_SEED, binom_bayes, block_sample, p_val,
                                  set_for_boot, word_like, words)
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import build_grid_axes, est_surfaces_on_grid
from pyegret.utils.season import set_season_label

__all__ = [
    "wbt",
    "est_slice_surfaces",
    "pa_vector",
    "make_combo",
    "make_two_years_results",
]

_PROBS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


def _decimal_in_year(moment: pd.Timestamp, year: int) -> float:
    """``year`` plus the fraction of the year elapsed at ``moment``, as R's paVector.

    R divides by the seconds from January 1, 00:00 to December 31, 23:59, not
    to the next January 1.
    """
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59)
    elapsed = (moment - start).total_seconds()
    span = (end - start).total_seconds()
    return year + elapsed / span


def pa_vector(year: int, pa_start: int, pa_long: int, vector_year) -> np.ndarray:
    """Grid steps inside one period of analysis. Mirrors R ``paVector()``.

    Returns:
        Zero-based indices into ``vector_year``.
    """
    if pa_start + pa_long > 13:
        # The period crosses January 1.
        min_year = year - 1
        max_time = (pd.Timestamp(year=year, month=pa_start + pa_long - 12, day=1)
                    - pd.Timedelta(days=1))
    else:
        min_year = year
        if pa_start + pa_long <= 12:
            max_time = (pd.Timestamp(year=year, month=pa_start + pa_long, day=1)
                        - pd.Timedelta(days=1))
        else:
            max_time = pd.Timestamp(year=year, month=12, day=31)
    min_time = pd.Timestamp(year=min_year, month=pa_start, day=1)
    min_dec = _decimal_in_year(min_time, min_year)
    max_dec = _decimal_in_year(max_time, year)
    vector_year = np.asarray(vector_year, dtype=float)
    return np.flatnonzero((vector_year >= min_dec) & (vector_year <= max_dec))


def est_slice_surfaces(elist: dict, year: int) -> np.ndarray:
    """The surface for one period of analysis only. Mirrors R ``estSliceSurfacesSimpleAlt()``.

    The full grid is returned, NaN except for the year steps of ``year``'s
    period of analysis plus one step either side. The settings come from
    ``INFO`` (see :func:`~pyegret.ci.bootstrap.set_for_boot`); the minimums
    are capped at what the Sample holds; and, as in R, the record ends used
    for edge adjustment are the first and last day of Daily.

    Returns:
        Array ``[year, logQ, 3]`` on the full grid.
    """
    info = elist["INFO"]
    sample = elist["Sample"]
    pa_start, pa_long = get_pa(elist)
    vector_year, vector_logQ = build_grid_axes(elist)

    min_num_uncen = min(float(info["minNumUncen"].iloc[0]), float(np.sum(sample["Uncen"])))
    min_num_obs = min(float(info["minNumObs"].iloc[0]), float(len(sample)))
    edge_adjust = True
    if "edgeAdjust" in info.columns:
        edge_adjust = bool(info["edgeAdjust"].iloc[0])

    inside = pa_vector(year, pa_start, pa_long, vector_year)
    # Tack on one grid step either side, as R does.
    indices = np.concatenate([[inside[0] - 1], inside, [inside[-1] + 1]])
    indices = indices[(indices >= 0) & (indices < len(vector_year))]

    daily_dec = elist["Daily"]["DecYear"].to_numpy(dtype=float)
    slice_surfaces, _ = est_surfaces_on_grid(
        sample, vector_year[indices], vector_logQ,
        window_y=float(info["windowY"].iloc[0]), window_q=float(info["windowQ"].iloc[0]),
        window_s=float(info["windowS"].iloc[0]), min_num_obs=min_num_obs,
        min_num_uncen=min_num_uncen, edge_adjust=edge_adjust, verbose=False,
        dec_low=daily_dec[0], dec_high=daily_dec[-1],
    )
    surfaces = np.full((len(vector_year), len(vector_logQ), 3), np.nan)
    surfaces[indices] = slice_surfaces
    return surfaces


def make_combo(surfaces1: np.ndarray, surfaces2: np.ndarray) -> np.ndarray:
    """Join two slice surfaces into one. Mirrors R ``makeCombo()``."""
    first = np.nan_to_num(surfaces1, nan=0.0)
    second = np.nan_to_num(surfaces2, nan=0.0)
    combo = first + second
    combo[combo == 0] = np.nan
    return combo


def make_two_years_results(elist: dict, combo: np.ndarray) -> np.ndarray:
    """Flow-normalized concentration and flux of the two years. Mirrors R ``makeTwoYearsResults()``.

    Returns:
        ``[FNConc year 1, FNConc year 2, FNFlux year 1, FNFlux year 2]``, flux
        in kg/day.

    Raises:
        ValueError: If the combined surface does not give exactly two
            complete years.
    """
    pa_start, pa_long = get_pa(elist)
    vector_year, vector_logQ = build_grid_axes(elist)
    daily = est_daily_from_surfaces(elist, surfaces=combo, vector_year=vector_year,
                                    vector_logQ=vector_logQ, daily=elist["Daily"])
    annual = setup_years(daily, pa_long=pa_long, pa_start=pa_start)
    conc = annual["FNConc"].to_numpy(dtype=float)
    flux = annual["FNFlux"].to_numpy(dtype=float)
    conc = conc[~np.isnan(conc)]
    flux = flux[~np.isnan(flux)]
    if len(conc) != 2 or len(flux) != 2:
        raise ValueError("The two slices did not give exactly two complete years.")
    return np.concatenate([conc, flux])


def _fmt(value: float, digits: int = 3, width: int = 7) -> str:
    """Roughly R's format(x, digits = d, width = w)."""
    text = f"{value:.{digits}g}"
    return f"{text:>{width}}"


def wbt(elist: dict, case_set_up: pd.DataFrame, start_seed: int = DEFAULT_START_SEED,
        verbose: bool = True, file_name: str | None = None) -> dict:
    """The WRTDS Bootstrap Test. Mirrors R ``wBT()``.

    Estimates the change in flow-normalized concentration and flux between
    ``year1`` and ``year2`` of ``case_set_up`` (stationary flow
    normalization), then repeats the estimate on block-bootstrap resamples
    until the direction of both trends is clear (judged with a Bayesian
    binomial interval, ``confStop`` 0.7) or ``nBoot`` replicates have
    succeeded. Replicate *i* uses R's seed ``start_seed + i``.

    EGRETci now recommends ``run_pairs`` with ``run_pairs_boot`` instead.

    Args:
        elist: eList dict. The windows and minimums are read from INFO when
            present (see :func:`~pyegret.ci.bootstrap.set_for_boot`).
        case_set_up: From :func:`~pyegret.ci.bootstrap.trend_set_up`.
        start_seed: Base seed. Default 494817, EGRETci's.
        verbose: Print the running table and the summary, as R does.
        file_name: Also write the summary to this text file (R's
            ``saveOutput``).

    Returns:
        Dict with ``bootOut`` (one-row DataFrame: ``rejectC``, ``pValC``,
        ``estC``, the 90/50/95 % intervals, ``likeCUp``, ``likeCDown``, the same
        for flux, ``baseConc``, ``baseFlux`` (10^6 kg/yr), ``iBoot``,
        ``startSeed``, ``nBootGood``), ``wordsOut``, the replicates ``xConc``,
        ``xFlux``, ``pConc``, ``pFlux`` (length ``iBoot``, NaN where a
        replicate failed), and ``year1``, ``year2``.
    """
    message = "run_pairs/run_pairs_boot is recommended over the wbt function."
    print(message)
    elist = set_for_boot(elist)
    info = elist["INFO"]
    case = case_set_up.iloc[0]
    year1 = int(case["year1"])
    year2 = int(case["year2"])
    n_boot = int(case["nBoot"])
    boot_break = int(case["bootBreak"])
    block_length = int(case["blockLength"])
    conf_stop = float(case["confStop"])
    pa_start, pa_long = get_pa(elist)

    surfaces1 = est_slice_surfaces(elist, year1)
    surfaces2 = est_slice_surfaces(elist, year2)
    res = make_two_years_results(elist, make_combo(surfaces1, surfaces2))
    reg_delta_conc = res[1] - res[0]
    base_conc = res[0]
    log_conc_diff = np.log(res[1]) - np.log(res[0])
    reg_delta_flux = (res[3] - res[2]) * KG_DAY_TO_MILLION_KG_YEAR
    base_flux = res[2] * KG_DAY_TO_MILLION_KG_YEAR
    log_flux_diff = np.log(res[3]) - np.log(res[2])

    lines = []
    short_name = str(info["shortName"].iloc[0]) if "shortName" in info.columns else ""
    param_name = str(info["paramShortName"].iloc[0]) if "paramShortName" in info.columns else ""
    period_name = set_season_label(pa_start, pa_long)
    lines.append(f"\n {short_name}   {param_name}")
    lines.append(f"\n {period_name}")
    if pa_start == 1 and pa_long == 12:
        lines.append(f"\n  Bootstrap process, for change from Calendar Year {year1}  to  {year2}")
    elif pa_start == 10 and pa_long == 12:
        lines.append(f"\n  Bootstrap process, for change from Water Year  {year1}  to Water Year  {year2}")
    else:
        lines.append(f"\n  Bootstrap process, for change from  {year1}  to  {year2} : {period_name}")
    lines.append(f"                   data set runs from Water Year {int(case['yearData1'])} "
                 f"to Water Year {int(case['yearData2'])}")
    lines.append(f"  Bootstrap block length in days {block_length}")
    lines.append(f"  bootBreak is {boot_break}  confStop is {conf_stop}")
    lines.append(f"\n WRTDS estimated concentration change is {_fmt(reg_delta_conc)}  mg/L")
    lines.append(f" WRTDS estimated flux change is         {_fmt(reg_delta_flux, 4, 8)}  10^6 kg/yr")
    if verbose:
        print("\n".join(lines))
        print(" nPos is cumulative number of positive trends")
        print("\n post_p is posterior mean estimate of probability of a positive trend")
        print(" Lower and Upper are estimates of the 90% CI values for magnitude of trend")
        print("\n      rep              Concentration             |              Flux")
        print("          value     nPos post_p   Lower   Upper  |     value   nPos  post_p    Lower   Upper")

    x_conc = np.full(2 * n_boot, np.nan)
    x_flux = np.full(2 * n_boot, np.nan)
    p_conc = np.full(2 * n_boot, np.nan)
    p_flux = np.full(2 * n_boot, np.nan)
    pos_conc = 0
    pos_flux = 0
    n_good = 0
    i_boot = 0
    quant_conc = np.full(len(_PROBS), np.nan)
    quant_flux = np.full(len(_PROBS), np.nan)
    for i_boot in range(1, 2 * n_boot + 1):
        boot_sample = block_sample(elist["Sample"], block_length, start_seed + i_boot)
        boot = dict(elist)
        boot["Sample"] = boot_sample
        try:
            boot_res = make_two_years_results(
                boot, make_combo(est_slice_surfaces(boot, year1), est_slice_surfaces(boot, year2)))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        n_good = n_good + 1
        k = n_good - 1
        x_conc[k] = (2 * reg_delta_conc) - (boot_res[1] - boot_res[0])
        x_flux[k] = (2 * reg_delta_flux) - ((boot_res[3] - boot_res[2]) * KG_DAY_TO_MILLION_KG_YEAR)
        log_conc = (2 * log_conc_diff) - (np.log(boot_res[1]) - np.log(boot_res[0]))
        p_conc[k] = 100 * np.exp(log_conc) - 100
        log_flux = (2 * log_flux_diff) - (np.log(boot_res[3]) - np.log(boot_res[2]))
        p_flux[k] = 100 * np.exp(log_flux) - 100

        if x_conc[k] > 0:
            pos_conc = pos_conc + 1
        interval_conc = binom_bayes(pos_conc, n_good, conf_stop)
        below_conc = interval_conc["upper"] < 0.05
        above_conc = interval_conc["lower"] > 0.95
        mid_conc = interval_conc["lower"] > 0.05 and interval_conc["upper"] < 0.95
        if x_flux[k] > 0:
            pos_flux = pos_flux + 1
        # R passes iBoot here, not nBootGood; kept as R has it.
        interval_flux = binom_bayes(pos_flux, i_boot, conf_stop)
        below_flux = interval_flux["upper"] < 0.05
        above_flux = interval_flux["lower"] > 0.95
        mid_flux = interval_flux["lower"] > 0.05 and interval_flux["upper"] < 0.95

        quant_conc = np.quantile(x_conc[:n_good], _PROBS, method="weibull")    # R type 6
        quant_flux = np.quantile(x_flux[:n_good], _PROBS, method="weibull")
        if verbose:
            print(f"{n_good:7d} {_fmt(x_conc[k])} {pos_conc:5d} {interval_conc['mean']:.3g} "
                  f"{_fmt(quant_conc[1])} {_fmt(quant_conc[7])}  |  {_fmt(x_flux[k], 4, 8)} "
                  f"{pos_flux:5d} {_fmt(interval_flux['mean'])} {_fmt(quant_flux[1], 4, 8)} "
                  f"{_fmt(quant_flux[7], 4, 8)}")

        clear_conc = below_conc or above_conc or mid_conc
        clear_flux = below_flux or above_flux or mid_flux
        test1 = clear_conc and clear_flux and n_good >= boot_break and i_boot > 30
        test2 = mid_conc and mid_flux and n_good >= boot_break and n_good <= 30
        if test1 or test2:
            break
        if n_good >= n_boot:
            break

    summary = []
    if i_boot == 2 * n_boot:
        summary.append(f"{i_boot} iterations were run. They only achieved {n_good} sucessful runs.")
    elif i_boot > n_boot:
        summary.append(f"It took {i_boot} iterations to achieve {n_boot} sucessful runs.")

    reject_c = bool(quant_conc[1] * quant_conc[7] > 0)
    reject_f = bool(quant_flux[1] * quant_flux[7] > 0)
    p_val_c = p_val(x_conc)
    p_val_f = p_val(x_flux)
    like_c_up = (pos_conc + 0.5) / (n_good + 1)
    like_f_up = (pos_flux + 0.5) / (n_good + 1)

    summary.append(f"\nShould we reject Ho that Flow Normalized Concentration Trend = 0 ? {words(reject_c)}")
    summary.append(f" best estimate is {_fmt(reg_delta_conc)} mg/L")
    summary.append(f"  Lower and Upper 90% CIs {_fmt(quant_conc[1], 3, 8)} {_fmt(quant_conc[7], 3, 8)}")
    summary.append(f" also 95% CIs {_fmt(quant_conc[0], 3, 8)} {_fmt(quant_conc[8], 3, 8)}")
    summary.append(f" and 50% CIs {_fmt(quant_conc[3], 3, 8)} {_fmt(quant_conc[5], 3, 8)}")
    summary.append(f" approximate two-sided p-value for Conc {_fmt(p_val_c, 2, 9)}")
    if pos_conc == 0 or pos_conc == n_good:
        summary.append("* Note p-value should be considered to be < stated value")
    summary.append(f" Likelihood that Flow Normalized Concentration is trending up = "
                   f"{_fmt(like_c_up, 3, 10)}  is trending down = {_fmt(1 - like_c_up, 3, 10)}")
    summary.append(f"\nShould we reject Ho that Flow Normalized Flux Trend = 0 ? {words(reject_f)}")
    summary.append(f" best estimate is {_fmt(reg_delta_flux, 4, 8)} 10^6 kg/year")
    summary.append(f"  Lower and Upper 90% CIs {_fmt(quant_flux[1], 3, 8)} {_fmt(quant_flux[7], 3, 8)}")
    summary.append(f" also 95% CIs {_fmt(quant_flux[0], 3, 8)} {_fmt(quant_flux[8], 3, 8)}")
    summary.append(f" and 50% CIs {_fmt(quant_flux[3], 3, 8)} {_fmt(quant_flux[5], 3, 8)}")
    summary.append(f" approximate two-sided p-value for Flux {_fmt(p_val_f, 2, 9)}")
    if pos_flux == 0 or pos_flux == n_good:
        summary.append("* Note p-value should be considered to be < stated value")
    summary.append(f" Likelihood that Flow Normalized Flux is trending up = "
                   f"{like_f_up:.3g}  is trending down= {1 - like_f_up:.3g}")
    words_out = word_like([like_c_up, 1 - like_c_up, like_f_up, 1 - like_f_up])
    summary.append(f"\n {words_out[0]}\n {words_out[2]}")
    summary.append(f" {words_out[1]}\n {words_out[3]}")
    if verbose:
        print("\n".join(summary))
    if file_name:
        with open(file_name, "w") as handle:
            handle.write("\n".join(lines + summary) + "\n")

    boot_out = pd.DataFrame([{
        "rejectC": reject_c, "pValC": p_val_c, "estC": reg_delta_conc,
        "lowC": quant_conc[1], "upC": quant_conc[7],
        "lowC50": quant_conc[3], "upC50": quant_conc[5],
        "lowC95": quant_conc[0], "upC95": quant_conc[8],
        "likeCUp": like_c_up, "likeCDown": 1 - like_c_up,
        "rejectF": reject_f, "pValF": p_val_f, "estF": reg_delta_flux,
        "lowF": quant_flux[1], "upF": quant_flux[7],
        "lowF50": quant_flux[3], "upF50": quant_flux[5],
        "lowF95": quant_flux[0], "upF95": quant_flux[8],
        "likeFUp": like_f_up, "likeFDown": 1 - like_f_up,
        "baseConc": base_conc, "baseFlux": base_flux,
        "iBoot": i_boot, "startSeed": start_seed, "nBootGood": n_good,
    }])
    return {"bootOut": boot_out, "wordsOut": words_out,
            "xConc": x_conc[:i_boot], "xFlux": x_flux[:i_boot],
            "pConc": p_conc[:i_boot], "pFlux": p_flux[:i_boot],
            "year1": year1, "year2": year2}

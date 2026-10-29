"""
Bootstrap confidence intervals and p-values for ``run_pairs`` and ``run_groups``.

Mirrors R EGRETci 2.0.5 ``runPairsBoot()``, ``runGroupsBoot()`` and the
functions they share (``run_bootstraps()``, ``single_boot_run()``,
``calc_boot_out()``, ``boot_message()``, ``check_pair_dates()``).

Each replicate resamples the Sample in blocks (:func:`~pyegret.ci.bootstrap.block_sample`,
seeded ``start_seed + i`` as in R, so replicate *i* uses the same rows as
EGRETci's), refits the model exactly as the original ``run_pairs`` or
``run_groups`` call did, and records the change it finds. The spread of those
changes, reflected about the original estimate, gives the confidence
intervals and the two-sided p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.analysis.run_groups import _group_mean
from pyegret.analysis.run_pairs import _between, _mean_fn
from pyegret.analysis.setup_years import setup_years, start_end
from pyegret.analysis.trend_setup import estimate_span_surface
from pyegret.ci.bootstrap import (DEFAULT_START_SEED, block_sample, p_val, word_like,
                                  words)
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import build_grid_axes
from pyegret.model.flex_fn import flex_fn
from pyegret.model.wall import stitch_surfaces
from pyegret.utils.season import set_season_label_by_user

__all__ = [
    "run_pairs_boot",
    "run_groups_boot",
    "single_boot_run",
    "calc_boot_out",
    "boot_message",
]

# kg/day to 10^6 kg/yr, as EGRETci writes it.
KG_DAY_TO_MILLION_KG_YEAR = 0.00036525

# The quantiles EGRETci reports, and their names.
_PROBS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


def _results_table(type_results: dict) -> pd.DataFrame:
    """The pairResults or groupResults table inside a run_pairs/run_groups result."""
    if "pairResults" in type_results:
        return type_results["pairResults"]
    return type_results["groupResults"]


def check_pair_dates(elist: dict, pairs: dict) -> None:
    """Raise if either year lies outside the Sample. Mirrors R ``check_pair_dates()``."""
    year_pair = pairs["yearPair"]
    span1 = start_end(year_pair["paStart"], year_pair["paLong"], year_pair["year1"])
    span2 = start_end(year_pair["paStart"], year_pair["paLong"], year_pair["year2"])
    dates = pd.to_datetime(elist["Sample"]["Date"])
    if span2["startDate"] > dates.max():
        raise ValueError("year2 is outside the Sample range")
    if span1["endDate"] < dates.min():
        raise ValueError("year1 is outside the Sample range")


def _boot_elist(elist: dict, boot_sample: pd.DataFrame) -> dict:
    """The eList with the resampled Sample and no fitted surface (R ``as.egret(..., NA)``)."""
    boot = {"INFO": elist["INFO"], "Daily": elist["Daily"], "Sample": boot_sample,
            "surfaces": None, "vector_year": None, "vector_logQ": None}
    return boot


def _pair_means(elist: dict, boot_sample: pd.DataFrame, pairs: dict) -> tuple:
    """Flow-normalized means of the two years from one resample (R's pair branch)."""
    year_pair = pairs["yearPair"]
    pa_start = year_pair["paStart"]
    pa_long = year_pair["paLong"]
    date_info = pairs["dateInfo"]
    blocks = pairs["SampleBlocks"]
    other = pairs["Other"]

    daily = elist["Daily"]
    daily1 = _between(daily, date_info["flowNormStart"].iloc[0], date_info["flowNormEnd"].iloc[0])
    daily2 = _between(daily, date_info["flowNormStart"].iloc[1], date_info["flowNormEnd"].iloc[1])
    span1 = start_end(pa_start, pa_long, year_pair["year1"])
    span2 = start_end(pa_start, pa_long, year_pair["year2"])
    sample1 = _between(boot_sample, blocks["sample1StartDate"], blocks["sample1EndDate"])
    sample2 = _between(boot_sample, blocks["sample2StartDate"], blocks["sample2EndDate"])

    boot = _boot_elist(elist, boot_sample)
    # R passes the minimums and edgeAdjust from the run_pairs result and leaves
    # the windows at estSurfaces' defaults; PyEGRET passes the windows the
    # run_pairs call used, which are the same unless they were changed.
    settings = {"window_y": other["windowY"], "window_q": other["windowQ"],
                "window_s": other["windowS"], "min_num_obs": other["minNumObs"],
                "min_num_uncen": other["minNumUncen"], "edge_adjust": other["edgeAdjust"],
                "verbose": False}
    surface1 = estimate_span_surface(boot, sample1, span1["startDate"], span1["endDate"],
                                     **settings)
    surface2 = estimate_span_surface(boot, sample2, span2["startDate"], span2["endDate"],
                                     **settings)
    est1 = est_daily_from_surfaces(boot, surfaces=surface1["surfaces"],
                                   vector_year=surface1["vector_year"],
                                   vector_logQ=surface1["vector_logQ"], daily=daily1)
    est2 = est_daily_from_surfaces(boot, surfaces=surface2["surfaces"],
                                   vector_year=surface2["vector_year"],
                                   vector_logQ=surface2["vector_logQ"], daily=daily2)
    c11, f11 = _mean_fn(est1, pa_long, pa_start)
    c22, f22 = _mean_fn(est2, pa_long, pa_start)
    return c11, f11, c22, f22


def _group_means(elist: dict, boot_sample: pd.DataFrame, groups: dict) -> tuple:
    """Flow-normalized group means from one resample (R's group branch)."""
    info = groups["groupInfo"]
    blocks = groups["SampleBlocks"]
    other = groups["Other"]

    boot = _boot_elist(elist, boot_sample)
    vector_year, vector_logQ = build_grid_axes(boot)
    windows = {"window_y": other["windowY"], "window_q": other["windowQ"],
               "window_s": other["windowS"], "min_num_obs": other["minNumObs"],
               "min_num_uncen": other["minNumUncen"], "edge_adjust": other["edgeAdjust"],
               "verbose": False}
    if other["wall"]:
        # As in R, stitch() relaxes the minimums from the resample's own split.
        surfaces = stitch_surfaces(boot, blocks["sample1EndDate"],
                                   surface_start=blocks["surfaceStart"],
                                   surface_end=blocks["surfaceEnd"], **windows)
        boot["surfaces"] = surfaces
        boot["vector_year"] = vector_year
        boot["vector_logQ"] = vector_logQ
    else:
        fitted = estimate_span_surface(boot, boot_sample, blocks["surfaceStart"],
                                       blocks["surfaceEnd"], **windows)
        boot["surfaces"] = fitted["surfaces"]
        boot["vector_year"] = fitted["vector_year"]
        boot["vector_logQ"] = fitted["vector_logQ"]

    daily_flex = flex_fn(boot, date_info=groups["dateInfo"])
    annual = setup_years(daily_flex, pa_long=info["paLong"], pa_start=info["paStart"])
    c11 = _group_mean(annual, info["group1firstYear"], info["group1lastYear"], "FNConc")
    f11 = _group_mean(annual, info["group1firstYear"], info["group1lastYear"], "FNFlux")
    c22 = _group_mean(annual, info["group2firstYear"], info["group2lastYear"], "FNConc")
    f22 = _group_mean(annual, info["group2firstYear"], info["group2lastYear"], "FNFlux")
    return c11, f11, c22, f22


def single_boot_run(i_boot: int, start_seed: int, elist: dict, type_results: dict,
                    block_length: int, kind: str, verbose: bool = True) -> dict | None:
    """One bootstrap replicate. Mirrors R ``single_boot_run()``.

    Args:
        i_boot: Replicate number; the resample is seeded ``start_seed + i_boot``.
        start_seed: Base seed.
        elist: The eList the trend was computed from.
        type_results: The result of ``run_pairs`` or ``run_groups``.
        block_length: Block length in days.
        kind: ``"pair"`` or ``"group"``.
        verbose: Print the replicate's result, as R does.

    Returns:
        Dict with ``xConc``, ``xFlux`` (the reflected changes, mg/L and
        10^6 kg/yr) and ``pConc``, ``pFlux`` (the same in percent), or None
        if the model could not be fitted to this resample.
    """
    if kind not in ("pair", "group"):
        raise ValueError("kind must be 'pair' or 'group'")
    boot_sample = block_sample(elist["Sample"], block_length, start_seed + i_boot)
    try:
        if kind == "pair":
            c11, f11, c22, f22 = _pair_means(elist, boot_sample, type_results)
        else:
            c11, f11, c22, f22 = _group_means(elist, boot_sample, type_results)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None

    f11 = f11 * KG_DAY_TO_MILLION_KG_YEAR
    f22 = f22 * KG_DAY_TO_MILLION_KG_YEAR
    table = _results_table(type_results)
    x11_conc = table.loc["Conc", "x11"]
    x22_conc = table.loc["Conc", "x22"]
    x11_flux = table.loc["Flux", "x11"]
    x22_flux = table.loc["Flux", "x22"]
    reg_delta_conc = x22_conc - x11_conc
    reg_delta_flux = x22_flux - x11_flux

    x_conc = (2 * reg_delta_conc) - (c22 - c11)
    x_flux = (2 * reg_delta_flux) - (f22 - f11)
    p_conc = np.nan
    p_flux = np.nan
    if not np.isnan(x_conc) and not np.isnan(x_flux):
        log_conc = 2 * (np.log(x22_conc) - np.log(x11_conc)) - (np.log(c22) - np.log(c11))
        p_conc = 100 * np.exp(log_conc) - 100
        log_flux = 2 * (np.log(x22_flux) - np.log(x11_flux)) - (np.log(f22) - np.log(f11))
        p_flux = 100 * np.exp(log_flux) - 100
        if verbose:
            print(f"iBoot, xConc and xFlux {i_boot}: {round(x_conc, 4)} {round(x_flux, 4)}")
    return {"xConc": x_conc, "xFlux": x_flux, "pConc": p_conc, "pFlux": p_flux}


def _run_bootstraps(elist: dict, type_results: dict, block_length: int, start_seed: int,
                    n_boot: int, kind: str, verbose: bool) -> dict:
    """Run replicates until ``n_boot`` succeed, or ``2 * n_boot`` have been tried.

    Mirrors R ``run_bootstraps()`` (serial branch). R fits one more replicate
    after reaching ``n_boot`` and discards it; PyEGRET stops at ``n_boot``,
    which gives the same results without the extra fit.
    """
    collected = {"xConc": [], "xFlux": [], "pConc": [], "pFlux": []}
    n_good = 0
    i_boot = 0
    for i_boot in range(1, 2 * n_boot + 1):
        result = single_boot_run(i_boot, start_seed, elist, type_results,
                                 block_length, kind, verbose=verbose)
        if result is not None:
            n_good = n_good + 1
            for name in collected:
                collected[name].append(result[name])
        if n_good >= n_boot:
            break

    if n_good < n_boot:
        print(f"{i_boot} iterations were run. They only achieved {n_good} sucessful runs.")
    elif i_boot > n_boot:
        print(f"It took {i_boot} iterations to achieve {n_boot} sucessful runs.")

    out = {}
    for name, values in collected.items():
        array = np.asarray(values, dtype=float)
        out[name] = array[~np.isnan(array)]
    return out


def calc_boot_out(boot_values: dict, type_results: dict, n_boot: int, start_seed: int,
                  block_length: int) -> dict:
    """Summarize the replicates. Mirrors R ``calc_boot_out()``.

    Returns:
        Dict with ``bootOut`` (a one-row DataFrame: ``rejectC``, ``pValC``,
        ``estC``, the 90 % interval ``lowC``/``upC``, the 50 % and 95 %
        intervals, ``likeCUp``, ``likeCDown``, the same for flux, ``baseConc``,
        ``baseFlux``, ``nBoot``, ``startSeed``, ``blockLength``,
        ``nBootGood``), ``wordsOut`` (four sentences), and the replicates
        ``xConc``, ``xFlux``, ``pConc``, ``pFlux``.
    """
    x_conc = boot_values["xConc"]
    x_flux = boot_values["xFlux"]
    quant_conc = np.quantile(x_conc, _PROBS, method="weibull")    # R type 6
    quant_flux = np.quantile(x_flux, _PROBS, method="weibull")
    n_good = len(x_conc)

    like_c_up = (np.sum(x_conc > 0) + 0.5) / (n_good + 1)
    like_f_up = (np.sum(x_flux > 0) + 0.5) / (n_good + 1)
    table = _results_table(type_results)

    boot_out = pd.DataFrame([{
        "rejectC": bool(quant_conc[1] * quant_conc[7] > 0),
        "pValC": p_val(x_conc),
        "estC": table.loc["Conc", "x22"] - table.loc["Conc", "x11"],
        "lowC": quant_conc[1], "upC": quant_conc[7],
        "lowC50": quant_conc[3], "upC50": quant_conc[5],
        "lowC95": quant_conc[0], "upC95": quant_conc[8],
        "likeCUp": like_c_up, "likeCDown": 1 - like_c_up,
        "rejectF": bool(quant_flux[1] * quant_flux[7] > 0),
        "pValF": p_val(x_flux),
        "estF": table.loc["Flux", "x22"] - table.loc["Flux", "x11"],
        "lowF": quant_flux[1], "upF": quant_flux[7],
        "lowF50": quant_flux[3], "upF50": quant_flux[5],
        "lowF95": quant_flux[0], "upF95": quant_flux[8],
        "likeFUp": like_f_up, "likeFDown": 1 - like_f_up,
        "baseConc": table.loc["Conc", "x11"],
        "baseFlux": table.loc["Flux", "x11"],
        "nBoot": n_boot, "startSeed": start_seed, "blockLength": block_length,
        "nBootGood": n_good,
    }])
    words_out = word_like([like_c_up, 1 - like_c_up, like_f_up, 1 - like_f_up])
    return {"bootOut": boot_out, "wordsOut": words_out,
            "xConc": x_conc, "xFlux": x_flux,
            "pConc": boot_values["pConc"], "pFlux": boot_values["pFlux"]}


def _r_format(value: float, digits: int = 3) -> str:
    """A number as R's format(x, digits = d) prints it, near enough."""
    return f"{value:.{digits}g}"


def boot_message(elist: dict, type_results: dict, boot: dict, kind: str = "pair") -> None:
    """Print the bootstrap summary. Mirrors R ``boot_message()``."""
    out = boot["bootOut"].iloc[0]
    table = _results_table(type_results)
    info = elist["INFO"]
    short_name = str(info["shortName"].iloc[0]) if "shortName" in info.columns else ""
    param_name = (str(info["paramShortName"].iloc[0])
                  if "paramShortName" in info.columns else "")
    if kind == "pair":
        pa_start = type_results["yearPair"]["paStart"]
        pa_long = type_results["yearPair"]["paLong"]
    else:
        pa_start = type_results["groupInfo"]["paStart"]
        pa_long = type_results["groupInfo"]["paLong"]

    reg_delta_conc = table.loc["Conc", "x22"] - table.loc["Conc", "x11"]
    reg_delta_flux = table.loc["Flux", "x22"] - table.loc["Flux", "x11"]

    print(f"\n   {short_name} \n   {param_name}")
    print(f"   {set_season_label_by_user(pa_start, pa_long)} ")
    if kind == "pair":
        year_pair = type_results["yearPair"]
        print(f"\n Change estimates are for  {year_pair['year2']}  minus  {year_pair['year1']}")
    else:
        groups = type_results["groupInfo"]
        print(f"\n Change estimates for\n average of {groups['group2firstYear']}  through "
              f"{groups['group2lastYear']}  minus average of {groups['group1firstYear']}  "
              f"through {groups['group1lastYear']} ")
    if type_results["Other"]["wall"]:
        wall = pd.Timestamp(type_results["SampleBlocks"]["sample1EndDate"])
        print(f"\n Sample data set was partitioned with a wall at  {wall.date()} \n")

    print(f"\n\nShould we reject Ho that Flow Normalized Concentration Trend = 0 ? "
          f"{words(out['rejectC'])}")
    print(f" best estimate of change in concentration is {_r_format(reg_delta_conc)} mg/L")
    print(f"  Lower and Upper 90% CIs {out['lowC']:.6g} {out['upC']:.6g}")
    print(f" also 95% CIs {out['lowC95']:.6g} {out['upC95']:.6g} ")
    print(f" and 50% CIs {out['lowC50']:.6g} {out['upC50']:.6g}")
    print(f" approximate two-sided p-value for Conc {_r_format(out['pValC'], 2):>9}")
    n_up = int(np.sum(boot["xConc"] > 0))
    if n_up == 0 or n_up == out["nBootGood"]:
        print("* Note p-value should be considered to be < stated value")
    print(f" Likelihood that Flow Normalized Concentration is trending up = "
          f"{_r_format(out['likeCUp'])}  is trending down = {_r_format(out['likeCDown'])}")

    print(f"\n\nShould we reject Ho that Flow Normalized Flux Trend = 0 ? "
          f"{words(out['rejectF'])}")
    print(f" best estimate of change in flux is {_r_format(reg_delta_flux)} 10^6 kg/year")
    print(f"  Lower and Upper 90% CIs {out['lowF']:.6g} {out['upF']:.6g}")
    print(f" also 95% CIs {out['lowF95']:.6g} {out['upF95']:.6g} ")
    print(f" and 50% CIs {out['lowF50']:.6g} {out['upF50']:.6g}")
    print(f" approximate two-sided p-value for Flux {_r_format(out['pValF'], 2):>9}")
    n_up = int(np.sum(boot["xFlux"] > 0))
    if n_up == 0 or n_up == out["nBootGood"]:
        print("* Note p-value should be considered to be < stated value")
    print(f" Likelihood that Flow Normalized Flux is trending up = "
          f"{_r_format(out['likeFUp'])}  is trending down = {_r_format(out['likeFDown'])}")
    if out["nBootGood"] < out["nBoot"]:
        print(f" The number of good replicates in the bootstrap was  {out['nBootGood']} "
              f" out of the  {out['nBoot']} total")
    print(f"\n {boot['wordsOut'][0]}\n {boot['wordsOut'][2]}")
    print(f" {boot['wordsOut'][1]}\n {boot['wordsOut'][3]}")


def run_pairs_boot(elist: dict, pairs: dict, n_boot: int = 100,
                   start_seed: int = DEFAULT_START_SEED, block_length: int = 200,
                   verbose: bool = True) -> dict:
    """Bootstrap uncertainty for a ``run_pairs`` result. Mirrors R ``runPairsBoot()``.

    Each replicate resamples the Sample in blocks of ``block_length`` days
    and repeats the ``run_pairs`` calculation exactly as it was set up
    (period of analysis, generalized flow normalization, wall, flow break,
    date limits). The changes from ``n_boot`` successful replicates give the
    confidence intervals, a two-sided p-value, and likelihoods of an upward
    or downward trend, for flow-normalized concentration and flux.

    Replicate *i* is seeded ``start_seed + i`` with R's generator, so it
    resamples the same rows as EGRETci's replicate *i*; the results differ
    from R's only by the small differences between the two model fits.

    This is slow: every replicate fits two surfaces. ``n_boot = 100`` (the
    default, as in R) takes a few minutes for a record like the Choptank's.

    Args:
        elist: The eList ``pairs`` was computed from.
        pairs: The result of :func:`~pyegret.analysis.run_pairs`.
        n_boot: Number of successful replicates wanted. Up to ``2 * n_boot``
            are tried.
        start_seed: Seed of the first replicate's resample is
            ``start_seed + 1``. Default 494817, EGRETci's.
        block_length: Block length in days. Default 200.
        verbose: Print each replicate and the summary, as R does.

    Returns:
        Dict as :func:`calc_boot_out` describes, plus ``year1``, ``year2``,
        ``paStart`` and ``paLong``.
    """
    check_pair_dates(elist, pairs)
    values = _run_bootstraps(elist, pairs, block_length, start_seed, n_boot, "pair", verbose)
    boot = calc_boot_out(values, pairs, n_boot, start_seed, block_length)
    year_pair = pairs["yearPair"]
    boot["year1"] = year_pair["year1"]
    boot["year2"] = year_pair["year2"]
    boot["paStart"] = year_pair["paStart"]
    boot["paLong"] = year_pair["paLong"]
    if verbose:
        boot_message(elist, pairs, boot, kind="pair")
    return boot


def run_groups_boot(elist: dict, groups: dict, n_boot: int = 100,
                    start_seed: int = DEFAULT_START_SEED, block_length: int = 200,
                    verbose: bool = True) -> dict:
    """Bootstrap uncertainty for a ``run_groups`` result. Mirrors R ``runGroupsBoot()``.

    As :func:`run_pairs_boot`, for the change between two groups of years.
    Every replicate fits the whole surface (with the wall, if ``run_groups``
    used one), so it is slower than the pairs version.

    Args:
        elist: The eList ``groups`` was computed from.
        groups: The result of :func:`~pyegret.analysis.run_groups`.
        n_boot: Number of successful replicates wanted.
        start_seed: Base seed. Default 494817, EGRETci's.
        block_length: Block length in days. Default 200.
        verbose: Print each replicate and the summary, as R does.

    Returns:
        Dict as :func:`calc_boot_out` describes, plus the four group years and
        ``paStart``, ``paLong``.
    """
    values = _run_bootstraps(elist, groups, block_length, start_seed, n_boot, "group", verbose)
    boot = calc_boot_out(values, groups, n_boot, start_seed, block_length)
    for name in ("group1firstYear", "group1lastYear", "group2firstYear",
                 "group2lastYear", "paStart", "paLong"):
        boot[name] = groups["groupInfo"][name]
    if verbose:
        boot_message(elist, groups, boot, kind="group")
    return boot

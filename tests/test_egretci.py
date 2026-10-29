"""
pyegret.ci (the EGRETci translation) against results R EGRETci itself produced.

The reference files in tests/reference/egretci/ were computed with the R
sources of EGRET 3.0.12 and EGRETci 2.0.5 kept in reference/ (see the README
there). Because PyEGRET draws its bootstrap resamples with R's own
random-number generator, seeded as EGRETci seeds it, replicate *i* here
resamples exactly the same samples as replicate *i* in R, and the two can be
compared one for one. What differs is only the model fit (SciPy's optimizer
against R's survreg), so the replicates agree to a fraction of a percent.

Most tests run a handful of replicates; the full 100-replicate comparisons
are in docs/validation.md.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdata", reason="pip install rdata to load the Choptank reference eList")

from pyegret import merge_report, model_estimation, wrtds_kalman  # noqa: E402
from pyegret.analysis import run_groups, run_pairs  # noqa: E402
from pyegret.ci import (block_sample, boot_annual, ci_bands, gen_daily_boot,  # noqa: E402
                        make_annual_pi, make_daily_pi, make_month_pi, month_seq_to_dec,
                        p_val, run_groups_boot, run_pairs_boot, trend_set_up, wbt,
                        word_like)
from pyegret.ci.bootstrap import binom_bayes  # noqa: E402
from pyegret.ci.r_random import r_runif, r_sample_index, r_set_seed  # noqa: E402
from pyegret.ci.wbt import make_combo, pa_vector  # noqa: E402
from pyegret.model.est_surfaces import build_grid_axes  # noqa: E402

_REFERENCE = Path(__file__).parent / "reference" / "egretci"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))
from choptank_wrtds import load_choptank  # noqa: E402


def _reference(name):
    with open(_REFERENCE / name) as handle:
        return json.load(handle)


def _array(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _int_array(values):
    numbers = _array(values)
    return numbers.astype(int)


@pytest.fixture(scope="module")
def choptank():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info, daily, sample = load_choptank()
    return merge_report(info, daily, sample)


@pytest.fixture(scope="module")
def small():
    return _reference("small.json")


# --- R's random numbers ------------------------------------------------

def test_r_runif_matches_r_set_seed_42():
    values, _ = r_runif(r_set_seed(42), 5)
    expected = [0.914806043496355, 0.937075413297862, 0.286139534786344,
                0.830447626067325, 0.641745518893003]
    assert np.allclose(values, expected, atol=1e-9)


def test_r_runif_matches_r_set_seed_123_across_the_table_refill():
    values, _ = r_runif(r_set_seed(123), 1000)
    assert np.allclose(values[:3], [0.287577520124614, 0.788305135443807, 0.408976921811700],
                       atol=1e-9)
    assert len(values) == 1000      # crosses the 624-word refill


def test_r_sample_rounding_matches_r_before_3_6():
    """set.seed(42); sample(1:10) under RNGversion('3.5.0')."""
    state = r_set_seed(42)
    pool = list(range(1, 11))
    drawn = []
    for n in range(10, 0, -1):
        index, state = r_sample_index(state, n)
        drawn.append(pool[index])
        pool[index] = pool[n - 1]
    assert drawn == [10, 9, 3, 6, 4, 8, 5, 1, 2, 7]


# --- The bootstrap building blocks -------------------------------------

def test_block_sample_draws_exactly_rs_rows(choptank, small):
    resampled = block_sample(choptank["Sample"], 200, start_seed=494818)
    assert np.array_equal(resampled["Julian"].to_numpy(dtype=float),
                          _array(small["block_sample_seed_494818_length_200_julian"]))
    assert np.allclose(resampled["ConcHigh"].to_numpy(dtype=float),
                       _array(small["block_sample_seed_494818_length_200_conc"]))


def test_block_sample_other_seed_and_block_length(choptank, small):
    resampled = block_sample(choptank["Sample"], 150, start_seed=7)
    assert np.array_equal(resampled["Julian"].to_numpy(dtype=float),
                          _array(small["block_sample_seed_7_length_150_julian"]))


def test_block_sample_keeps_size_and_order(choptank):
    resampled = block_sample(choptank["Sample"], 200, start_seed=1)
    assert len(resampled) == len(choptank["Sample"])
    assert resampled["Julian"].is_monotonic_increasing


def test_p_val_matches_r(small):
    values = [-0.3, 0.2, 0.5, 0.1, -0.05, 0.7, 0.9, 0, np.nan, 0.4]
    ours = [p_val(values), p_val(np.abs(values)), p_val([1, 2, -1, 3, 4, 5, -2, 6])]
    assert np.allclose(ours, small["p_val"])


def test_binom_bayes_matches_r(small):
    table = small["binom_bayes_central_0.7"]
    for x, n, mean, lower, upper in zip(table["x"], table["n"], table["mean"],
                                        table["lower"], table["upper"]):
        ours = binom_bayes(int(x), int(n), 0.7)
        assert ours["mean"] == pytest.approx(mean)
        assert ours["lower"] == pytest.approx(lower, abs=1e-7)
        assert ours["upper"] == pytest.approx(upper, abs=1e-7)


def test_word_like_matches_r(small):
    assert word_like(small["word_like_inputs"]) == small["word_like"]


def test_make_combo_adds_slices_and_keeps_gaps_missing():
    first = np.array([[1.0, np.nan], [np.nan, np.nan]])
    second = np.array([[np.nan, np.nan], [2.0, np.nan]])
    combo = make_combo(first, second)
    assert combo[0, 0] == 1.0 and combo[1, 0] == 2.0
    assert np.isnan(combo[0, 1]) and np.isnan(combo[1, 1])


def test_pa_vector_covers_one_water_year(choptank):
    vector_year, _ = build_grid_axes(choptank)
    inside = pa_vector(1990, 10, 12, vector_year)
    years = vector_year[inside]
    assert years.min() >= 1989.74 and years.max() <= 1990.75
    assert len(inside) == 16


def test_month_seq_to_dec():
    assert month_seq_to_dec(1) == pytest.approx(1850 + 1 / 24)
    assert month_seq_to_dec(13) == pytest.approx(1851 + 1 / 24)


# --- run_pairs_boot ----------------------------------------------------

def test_run_pairs_boot_replicates_match_r_one_for_one(choptank):
    """The EGRETci vignette case: a wall, a flow break, April-August."""
    reference = _reference("pairs_wall_season.json")
    pairs = run_pairs(choptank, 1985, 2010, window_side=7, flow_break=True,
                      q1_end_date="1995-05-31", wall=True, sample1_end_date="1995-05-31",
                      q_start_date="1979-10-01", q_end_date="2010-09-30",
                      pa_start=4, pa_long=5, verbose=False)
    boot = run_pairs_boot(choptank, pairs, n_boot=6, verbose=False)
    for name in ("xConc", "xFlux", "pConc"):
        theirs = _array(reference[name])[:6]
        assert np.allclose(boot[name], theirs, rtol=0.003), name


def test_run_pairs_boot_summary_has_rs_fields(choptank):
    pairs = run_pairs(choptank, 1985, 2010, window_side=7, verbose=False)
    boot = run_pairs_boot(choptank, pairs, n_boot=4, verbose=False)
    reference = _reference("pairs_simple.json")
    assert list(boot["bootOut"].columns) == list(reference["bootOut"].keys())
    assert np.allclose(boot["xConc"], _array(reference["xConc"])[:4], rtol=0.005)
    out = boot["bootOut"].iloc[0]
    assert out["nBootGood"] == 4
    assert out["lowC"] <= out["upC"]
    assert boot["year1"] == 1985 and boot["year2"] == 2010


def test_run_pairs_boot_prints_rs_summary(choptank, capsys):
    pairs = run_pairs(choptank, 1985, 2010, window_side=7, verbose=False)
    run_pairs_boot(choptank, pairs, n_boot=2, verbose=True)
    printed = capsys.readouterr().out
    assert "iBoot, xConc and xFlux 1:" in printed
    assert "Should we reject Ho that Flow Normalized Concentration Trend = 0 ?" in printed
    assert "Upward trend in concentration is" in printed


# --- wbt ---------------------------------------------------------------

def test_wbt_replicates_match_r(choptank):
    reference = _reference("wbt.json")
    info = choptank["INFO"].copy()
    info["windowY"] = 10.0
    info["windowQ"] = 2.0
    info["windowS"] = 0.5
    info["minNumObs"] = 100
    info["minNumUncen"] = 50
    elist = dict(choptank)
    elist["INFO"] = info
    case = trend_set_up(elist, 1985, 2005, n_boot=5, boot_break=5)
    result = wbt(elist, case, verbose=False)
    assert result["bootOut"]["estC"].iloc[0] == pytest.approx(
        reference["bootOut"]["estC"], rel=0.002)
    assert np.allclose(result["xConc"][:5], _array(reference["xConc"])[:5], rtol=0.004)
    assert np.allclose(result["xFlux"][:5], _array(reference["xFlux"])[:5], rtol=0.007)


def test_trend_set_up_checks_the_years(choptank):
    with pytest.raises(ValueError, match="year1"):
        trend_set_up(choptank, 1970, 2005)
    case = trend_set_up(choptank, 1985, 2005, n_boot=20, boot_break=39)
    assert case["bootBreak"].iloc[0] == 20      # capped at nBoot, as in R


# --- run_groups_boot ---------------------------------------------------

def test_run_groups_boot_matches_r(choptank):
    reference = _reference("groups_wall.json")
    groups = run_groups(choptank, 7, 1995, 2004, 2005, 2010, wall=True,
                        sample1_end_date="2004-10-30", pa_start=4, pa_long=2, verbose=False)
    boot = run_groups_boot(choptank, groups, n_boot=2, verbose=False)
    # The group changes are small, some near zero, so compare in absolute terms
    # (mg/L and 10^6 kg/yr); over 20 replicates the worst differences in R's
    # runs were 0.0016 and 0.0005.
    assert np.allclose(boot["xConc"], _array(reference["xConc"])[:2], rtol=0, atol=0.003)
    assert np.allclose(boot["xFlux"], _array(reference["xFlux"])[:2], rtol=0, atol=0.001)


# --- Confidence bands --------------------------------------------------

@pytest.fixture(scope="module")
def fitted(choptank):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model_estimation(choptank, verbose=False)


def test_boot_annual_matches_r(fitted):
    reference = _reference("ci_model.json")
    replicate = boot_annual(fitted, 200, 494817 + 1)
    theirs = reference["reps"][0]
    assert np.array_equal(replicate["year"].to_numpy(), _int_array(theirs["year"]))
    assert np.allclose(replicate["FNConc"], _array(theirs["FNConc"]), rtol=0.003)
    assert np.allclose(replicate["FNFlux"], _array(theirs["FNFlux"]), rtol=0.005)


def test_ci_bands_reflects_replicates_about_the_estimate(fitted):
    """With R's own replicates, the band follows R's formula exactly."""
    reference = _reference("ci_model.json")
    replicates = [pd.DataFrame({"year": _int_array(r["year"]),
                                "FNConc": _array(r["FNConc"]),
                                "FNFlux": _array(r["FNFlux"])})
                  for r in reference["reps"]]
    band = ci_bands(fitted, replicates)
    theirs = reference["ci"]
    # The baseline is PyEGRET's own fit, so allow for its small difference from R's.
    assert np.allclose(band["FNConcLow"], _array(theirs["FNConcLow"]), rtol=0.003)
    assert np.allclose(band["FNFluxHigh"], _array(theirs["FNFluxHigh"]), rtol=0.005)
    assert band.attrs["nBoot"] == 5
    assert np.all(band["FNConcLow"] <= band["FNConcHigh"])


# --- Prediction intervals ----------------------------------------------

@pytest.fixture(scope="module")
def daily_boot(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kalman = wrtds_kalman(fitted, seed=1, verbose=False)
        out = gen_daily_boot(kalman, n_boot=2, n_kalman=2, set_seed=1, verbose=False)
    return kalman, out


def test_gen_daily_boot_uses_each_refit(daily_boot):
    kalman, out = daily_boot
    assert out.shape == (len(kalman["Daily"]), 4)
    # Same Kalman seed, different refit: the records differ.
    assert not np.allclose(out[:, 0], out[:, 2])


def test_gen_daily_boot_can_reproduce_egretci_2_0_5(fitted):
    """EGRETci 2.0.5 ignores the refits, so equal seeds give equal records."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kalman = wrtds_kalman(fitted, seed=1, niter=2, verbose=False)
        out = gen_daily_boot(kalman, n_boot=2, n_kalman=1, set_seed=1,
                             boot_surfaces=False, verbose=False)
    assert np.allclose(out[:, 0], out[:, 1])


def test_prediction_interval_tables(daily_boot):
    kalman, out = daily_boot
    annual = make_annual_pi(out, kalman)
    monthly = make_month_pi(out, kalman)
    daily = make_daily_pi(out, kalman)
    for table in (annual["flux"], monthly["conc"], daily["flux"]):
        values = table[["p5", "p50", "p95"]].to_numpy()
        assert np.all(values[:, 0] <= values[:, 1] + 1e-12)
        assert np.all(values[:, 1] <= values[:, 2] + 1e-12)
    assert len(daily["flux"]) == len(kalman["Daily"])
    assert list(annual["flux"].columns[:3]) == ["DecYear", "p1", "p2.5"]


# --- A fidelity guard added with EGRETci -------------------------------

def test_surfaces_refuse_too_few_samples_as_r_does(choptank):
    """R runSurvReg stops when minNumObs >= the number of samples."""
    from pyegret.model.est_surfaces import est_surfaces
    few = dict(choptank)
    few["Sample"] = choptank["Sample"].iloc[:100]
    with pytest.raises(ValueError, match="min_num_obs"):
        est_surfaces(few, min_num_obs=100, min_num_uncen=50, verbose=False)

"""
run_groups, run_series, surface_start_end and WRTDS-Kalman, each checked
against what the R source in reference/EGRET_R computes.
"""
import numpy as np
import pandas as pd
import pytest

from pyegret.analysis.run_groups import run_groups
from pyegret.analysis.run_pairs import run_pairs
from pyegret.analysis.run_series import run_series
from pyegret.analysis.setup_years import setup_years
from pyegret.analysis.trend_setup import surface_start_end
from pyegret.model.kalman import gen_missing, precompute_l, random_subset, wrtds_kalman
from pyegret.model.model_estimation import model_estimation
from pyegret.testing import make_synthetic_elist, make_truth


@pytest.fixture(scope="module")
def flows_changed():
    return make_synthetic_elist(truth=make_truth(trend=0.0), flow_trend=0.03, seed=31)


@pytest.fixture(scope="module")
def modelled():
    elist = make_synthetic_elist(start="1995-01-01", end="2004-12-31", n_samples=200,
                                 censor_fraction=0.1, seed=2)
    return model_estimation(elist, verbose=False)


# ---------------------------------------------------------------------------
# surfaceStartEnd
# ---------------------------------------------------------------------------

def test_surface_start_end_water_years():
    span = surface_start_end(10, 12, "1990-01-15", "2009-12-20")
    assert span["surfaceStart"] == pd.Timestamp("1989-10-01")
    assert span["surfaceEnd"] == pd.Timestamp("2010-09-30")


def test_surface_start_end_on_a_boundary():
    # R's loop stops at the first period starting on or after date1 and then
    # steps back one, so a date exactly on a period start gets the period
    # before it too. Reproduced as R has it.
    span = surface_start_end(10, 12, "1990-10-01", "2009-09-30")
    assert span["surfaceStart"] == pd.Timestamp("1989-10-01")
    assert span["surfaceEnd"] == pd.Timestamp("2009-09-30")


# ---------------------------------------------------------------------------
# runSeries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series(flows_changed):
    return run_series(flows_changed, window_side=6, verbose=False)


def test_run_series_fills_the_surface_span(series):
    daily = series["Daily"]
    inside = (daily["Date"] >= "1990-01-01") & (daily["Date"] <= "2009-09-30")
    assert np.all(np.isfinite(daily.loc[inside, "FNConc"]))


def test_run_series_records_settings_in_info(series):
    info = series["INFO"]
    assert info["windowSide"].iloc[0] == 6
    assert info["paStart"].iloc[0] == 10
    assert info["surfaceStart"].iloc[0] == pd.Timestamp("1989-10-01")


def test_run_series_moving_window_follows_the_flow_trend(flows_changed, series):
    # The concentration-discharge relation is fixed and flows rise, so with a
    # moving window the flow-normalized concentration rises; with a fixed
    # window it barely moves.
    stationary = run_series(flows_changed, window_side=0, verbose=False)
    moving_annual = setup_years(series["Daily"])
    fixed_annual = setup_years(stationary["Daily"])
    moving_change = moving_annual["FNConc"].iloc[-1] / moving_annual["FNConc"].iloc[1] - 1
    fixed_change = fixed_annual["FNConc"].iloc[-1] / fixed_annual["FNConc"].iloc[1] - 1
    assert moving_change > fixed_change + 0.05


def test_run_series_does_not_change_the_input(flows_changed):
    assert "FNConc" not in flows_changed["Daily"].columns


# ---------------------------------------------------------------------------
# runGroups
# ---------------------------------------------------------------------------

def test_run_groups_decomposition_adds_up(flows_changed):
    result = run_groups(flows_changed, 6, 1992, 1996, 2004, 2008, verbose=False)
    for label in ("Conc", "Flux"):
        row = result["groupResults"].loc[label]
        assert row["CQTC"] + row["QTC"] == pytest.approx(row["TotalChange"])
        assert row["TotalChange"] == pytest.approx(row["x22"] - row["x11"])


def test_run_groups_flow_driven_change_lands_in_qtc(flows_changed):
    result = run_groups(flows_changed, 6, 1992, 1996, 2004, 2008, verbose=False)
    row = result["groupResults"].loc["Conc"]
    assert abs(row["QTC"]) > abs(row["CQTC"])


def test_run_groups_single_year_groups_match_run_pairs_direction(flows_changed):
    # Groups of one year approximate a pair; the signs must agree.
    groups = run_groups(flows_changed, 6, 1992, 1992.5, 2008, 2008.5, verbose=False)
    pairs = run_pairs(flows_changed, 1992, 2008, window_side=6, verbose=False)
    g = groups["groupResults"].loc["Conc", "TotalChange"]
    p = pairs["pairResults"].loc["Conc", "TotalChange"]
    assert np.sign(g) == np.sign(p)


def test_run_groups_stationary_has_no_qtc(flows_changed):
    result = run_groups(flows_changed, 0, 1992, 1996, 2004, 2008, verbose=False)
    assert result["groupResults"].loc["Conc", "QTC"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# WRTDS-Kalman
# ---------------------------------------------------------------------------

def test_gen_missing_keeps_the_end_points():
    l_full = precompute_l(0.9, 50)
    rng = np.random.default_rng(0)
    segment = gen_missing(1.5, -0.5, 0.9, 10, l_full, rng)
    assert len(segment) == 10
    assert segment[0] == 1.5
    assert segment[-1] == -0.5


def test_gen_missing_matches_the_ar1_bridge_mean():
    # The conditional mean of an AR(1) bridge, by brute force: invert the
    # covariance of the whole segment and condition on the two ends.
    rho, n = 0.8, 7
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    cov = rho ** lag
    ends = [0, n - 1]
    inner = list(range(1, n - 1))
    x_ends = np.array([2.0, -1.0])
    expected = cov[np.ix_(inner, ends)] @ np.linalg.solve(cov[np.ix_(ends, ends)], x_ends)

    l_full = precompute_l(rho, 50)
    rng = np.random.default_rng(1)
    draws = np.array([gen_missing(2.0, -1.0, rho, n, l_full, rng)[1:-1] for _ in range(20000)])
    assert np.allclose(draws.mean(axis=0), expected, atol=0.02)


def test_random_subset_keeps_one_row_per_day():
    df = pd.DataFrame({"Julian": [1, 2, 2, 2, 3], "v": [10, 20, 21, 22, 30]})
    out = random_subset(df, "Julian", seed=4)
    assert out["Julian"].tolist() == [1, 2, 3]
    assert out["v"].iloc[1] in (20, 21, 22)


@pytest.fixture(scope="module")
def kalman(modelled):
    return wrtds_kalman(modelled, niter=30, seed=5, verbose=False)


def test_kalman_reproduces_sampled_days(modelled, kalman):
    daily = kalman["Daily"]
    sample = modelled["Sample"]
    merged = pd.merge(daily, sample[["Date", "ConcAve", "Uncen"]], on="Date")
    detected = merged[merged["Uncen"] == 1]
    assert np.allclose(detected["GenConc"], detected["ConcAve"], rtol=1e-10)


def test_kalman_is_close_to_wrtds_on_average(kalman):
    daily = kalman["Daily"]
    ratio = np.nanmean(daily["GenConc"] / daily["ConcDay"])
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_kalman_flux_definition(kalman):
    daily = kalman["Daily"]
    assert np.allclose(daily["GenConc"], daily["GenFlux"] / (daily["Q"] * 86.4))


def test_kalman_seed_is_reproducible(modelled, kalman):
    again = wrtds_kalman(modelled, niter=30, seed=5, verbose=False)
    assert np.allclose(again["Daily"]["GenFlux"], kalman["Daily"]["GenFlux"])


def test_kalman_annual_results_include_gen_columns(kalman):
    annual = setup_years(kalman["Daily"])
    assert "GenConc" in annual.columns and "GenFlux" in annual.columns


def test_kalman_does_not_change_the_input(modelled, kalman):
    assert "GenConc" not in modelled["Daily"].columns


def test_kalman_needs_a_model():
    with pytest.raises(ValueError, match="model_estimation"):
        wrtds_kalman(make_synthetic_elist(seed=1))


def test_blank_time(kalman):
    from pyegret.elist import blank_time
    blanked = blank_time(kalman, "1999-10-01", "2000-09-30")
    daily = blanked["Daily"]
    inside = (daily["Date"] >= "1999-10-01") & (daily["Date"] <= "2000-09-30")
    for column in ("ConcDay", "FluxDay", "FNConc", "FNFlux", "GenConc", "GenFlux"):
        assert np.all(daily.loc[inside, column].isna())
        assert np.all(daily.loc[~inside, column].notna())
    annual = setup_years(daily)
    assert np.sum(annual["FNConc"].isna()) == 1
    assert np.all(kalman["Daily"]["ConcDay"].notna())


def test_counts(modelled):
    from pyegret.elist import n_censored_vals, n_discharge, n_observations
    assert n_observations(modelled) == len(modelled["Sample"])
    assert n_discharge(modelled) == len(modelled["Daily"])
    assert n_censored_vals(modelled) == int(np.sum((modelled["Sample"]["Uncen"] == 0)))

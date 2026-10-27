"""
Phase 4 tests — analysis functions.

Fixtures use *synthetic* data so no live network calls or expensive model
runs are needed.  The ``modeled_elist`` fixture manually injects
ConcDay / FluxDay / FNConc / FNFlux into Daily and yHat / ConcHat / resid
into Sample, mimicking the output of ``model_estimation``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyegret.elist import make_elist
from pyegret.data.populate import (
    populate_date_columns,
    populate_sample_columns,
    merge_daily_q,
)
from pyegret.analysis.setup_years import setup_years, set_pa, start_end
from pyegret.analysis.monthly_results import calculate_monthly_results
from pyegret.analysis.error_stats import error_stats, flux_bias_stat
from pyegret.analysis.table_change import table_change, table_results

_RNG = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_daily(n_years: int = 12) -> pd.DataFrame:
    """Synthetic Daily DataFrame covering n_years of daily data."""
    dates = pd.date_range("1990-01-01", periods=365 * n_years, freq="D")
    df = pd.DataFrame({"Date": dates})
    df["Q"] = np.clip(np.exp(_RNG.normal(1.5, 0.8, len(df))), 0.01, None)
    df["LogQ"] = np.log(df["Q"])
    df = populate_date_columns(df)
    return df.reset_index(drop=True)


def _make_sample(daily: pd.DataFrame, n: int = 240) -> pd.DataFrame:
    """Synthetic Sample with ~n observations drawn from the daily record."""
    idx = _RNG.choice(len(daily), size=n, replace=False)
    dates = daily.iloc[idx]["Date"].values
    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    df["ConcHigh"] = np.clip(np.exp(_RNG.normal(0.5, 0.6, n)), 0.005, None)
    df["ConcLow"] = df["ConcHigh"].copy()
    df["Uncen"] = 1
    # 15 % censored
    cens_mask = _RNG.uniform(size=n) < 0.15
    df.loc[cens_mask, "Uncen"] = 0
    df.loc[cens_mask, "ConcLow"] = 0.0
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)
    df = populate_date_columns(df)
    df = populate_sample_columns(df)
    df = merge_daily_q(daily, df)   # adds Q, LogQ
    return df


@pytest.fixture(scope="module")
def base_elist() -> dict:
    """eList with Daily + Sample but no model output."""
    daily = _make_daily(12)
    INFO = pd.DataFrame({"shortName": ["Test Site"], "param_cd": ["00631"]})
    sample = _make_sample(daily)
    return make_elist(INFO=INFO, Daily=daily, Sample=sample)


@pytest.fixture(scope="module")
def modeled_elist(base_elist: dict) -> dict:
    """eList with synthetic model output columns added (no actual model run)."""
    elist = make_elist(
        INFO=base_elist["INFO"].copy(),
        Daily=base_elist["Daily"].copy(),
        Sample=base_elist["Sample"].copy(),
    )
    nd = len(elist["Daily"])
    ns = len(elist["Sample"])

    # Inject model columns into Daily
    elist["Daily"]["ConcDay"] = np.clip(np.exp(_RNG.normal(0.5, 0.4, nd)), 0.001, None)
    elist["Daily"]["FluxDay"] = elist["Daily"]["ConcDay"] * elist["Daily"]["Q"] * 86.4
    elist["Daily"]["FNConc"]  = np.clip(np.exp(_RNG.normal(0.5, 0.3, nd)), 0.001, None)
    elist["Daily"]["FNFlux"]  = elist["Daily"]["FNConc"] * elist["Daily"]["Q"] * 86.4

    # Inject model columns into Sample
    conc_ave = elist["Sample"]["ConcAve"].values.clip(1e-10)
    yhat = np.log(conc_ave) + _RNG.normal(0.0, 0.2, ns)
    se   = np.abs(_RNG.normal(0.5, 0.1, ns))
    elist["Sample"]["yHat"]    = yhat
    elist["Sample"]["SE"]      = se
    elist["Sample"]["ConcHat"] = np.exp(yhat + 0.5 * se ** 2)
    elist["Sample"]["resid"]   = np.log(conc_ave) - yhat

    return elist


# ──────────────────────────────────────────────────────────────────────────────
# setup_years / set_pa / start_end
# ──────────────────────────────────────────────────────────────────────────────

# --- SetupYears (R setupYears semantics) ---------------------------------

def test_setup_years_columns_match_egret(modeled_elist):
    result = setup_years(modeled_elist["Daily"])
    assert list(result.columns) == ["DecYear", "Q", "Conc", "Flux", "FNConc",
                                    "FNFlux", "PeriodLong", "PeriodStart"]

def test_setup_years_without_model_has_only_flow(base_elist):
    result = setup_years(base_elist["Daily"])
    assert list(result.columns) == ["DecYear", "Q", "PeriodLong", "PeriodStart"]

def test_setup_years_accepts_an_elist(modeled_elist):
    from_dict = setup_years(modeled_elist)
    from_daily = setup_years(modeled_elist["Daily"])
    pd.testing.assert_frame_equal(from_dict, from_daily)

def test_setup_years_only_complete_water_years(base_elist):
    # Daily runs from 1990-01-01 to late December 2001, so the complete water
    # years are 1991 (Oct 1990 - Sep 1991) through 2001.
    result = setup_years(base_elist["Daily"], pa_long=12, pa_start=10)
    years = np.trunc(result["DecYear"].to_numpy() + 0.5)
    assert years[0] == 1991
    assert years[-1] == 2001
    assert len(result) == 11

def test_setup_years_calendar_years(base_elist):
    result = setup_years(base_elist["Daily"], pa_long=12, pa_start=1)
    assert len(result) == 12

def test_setup_years_mean_matches_daily(modeled_elist):
    daily = modeled_elist["Daily"]
    result = setup_years(daily, pa_long=12, pa_start=10)
    in_wy1995 = (daily["Date"] >= "1994-10-01") & (daily["Date"] <= "1995-09-30")
    row = np.argmin(np.abs(result["DecYear"].to_numpy() - 1995.25))
    assert result["Q"].iloc[row] == pytest.approx(daily.loc[in_wy1995, "Q"].mean())
    assert result["FNConc"].iloc[row] == pytest.approx(daily.loc[in_wy1995, "FNConc"].mean())

def test_setup_years_season(base_elist):
    result = setup_years(base_elist["Daily"], pa_long=6, pa_start=4)
    assert np.all((result["PeriodLong"] == 6))
    assert np.all((result["PeriodStart"] == 4))

def test_setup_years_rejects_gaps(base_elist):
    daily = base_elist["Daily"].drop(index=[100, 101])
    with pytest.raises(ValueError, match="gaps"):
        setup_years(daily)

def test_setup_years_needs_ninety_percent_of_days(modeled_elist):
    daily = modeled_elist["Daily"].copy()
    in_wy1995 = (daily["Date"] >= "1994-10-01") & (daily["Date"] <= "1995-09-30")
    rows = np.flatnonzero(in_wy1995.to_numpy())[:60]   # blank 60 of 365 days
    daily.loc[rows, "ConcDay"] = np.nan
    result = setup_years(daily)
    row = np.argmin(np.abs(result["DecYear"].to_numpy() - 1995.25))
    assert np.isnan(result["Conc"].iloc[row])
    assert np.isfinite(result["Q"].iloc[row])

def test_set_pa_stores_values_in_info(base_elist):
    updated = set_pa(base_elist, pa_start=4, pa_long=6)
    assert int(updated["INFO"]["paStart"].iloc[0]) == 4
    assert int(updated["INFO"]["paLong"].iloc[0]) == 6
    assert "paStart" not in base_elist["INFO"].columns

def test_start_end_water_year():
    span = start_end(10, 12, 2001)
    assert span["startDate"] == pd.Timestamp("2000-10-01")
    assert span["endDate"] == pd.Timestamp("2001-09-30")

def test_start_end_leap_february():
    span = start_end(12, 3, 2000)
    assert span["startDate"] == pd.Timestamp("1999-12-01")
    assert span["endDate"] == pd.Timestamp("2000-02-29")


# ──────────────────────────────────────────────────────────────────────────────
# calculate_monthly_results
# ──────────────────────────────────────────────────────────────────────────────

def test_monthly_results_columns_match_egret(modeled_elist):
    df = calculate_monthly_results(modeled_elist)
    assert list(df.columns) == ["Month", "Year", "nDays", "DecYear", "Q",
                                "Conc", "FNConc", "Flux", "FNFlux"]

def test_monthly_results_one_row_per_month(modeled_elist):
    df = calculate_monthly_results(modeled_elist)
    assert len(df) == len(np.unique(modeled_elist["Daily"]["MonthSeq"]))

def test_monthly_results_month_mean(modeled_elist):
    daily = modeled_elist["Daily"]
    df = calculate_monthly_results(modeled_elist)
    first = daily["MonthSeq"] == daily["MonthSeq"].iloc[0]
    assert df["Conc"].iloc[0] == pytest.approx(daily.loc[first, "ConcDay"].mean())
    assert df["Month"].iloc[0] == 1

def test_monthly_results_drops_short_months(modeled_elist):
    daily = modeled_elist["Daily"].copy()
    first = (daily["MonthSeq"] == daily["MonthSeq"].iloc[0]).to_numpy()
    rows = np.flatnonzero(first)[:20]
    daily.loc[rows, "ConcDay"] = np.nan
    df = calculate_monthly_results({"Daily": daily})
    assert df["DecYear"].iloc[0] > 1990.08


# ──────────────────────────────────────────────────────────────────────────────
# error_stats / flux_bias_stat
# ──────────────────────────────────────────────────────────────────────────────

# --- ErrorStats --------------------------------------------------------

def test_error_stats_columns_match_egret(modeled_elist):
    stats = error_stats(modeled_elist, seed=1, verbose=False)
    assert list(stats.columns) == ["RsqLogC", "RsqLogF", "rmse", "sepPercent"]
    assert len(stats) == 1

def test_error_stats_rmse_is_sd_of_randomised_residuals(modeled_elist):
    from pyegret.data.quality import make_augmented_sample
    stats = error_stats(modeled_elist, seed=1, verbose=False)
    augmented = make_augmented_sample(modeled_elist, seed=1)
    resid = augmented["Sample"]["rResid"].to_numpy()
    assert stats["rmse"].iloc[0] == pytest.approx(np.std(resid, ddof=1))

def test_error_stats_sep_percent_formula(modeled_elist):
    stats = error_stats(modeled_elist, seed=1, verbose=False)
    rmse = stats["rmse"].iloc[0]
    assert stats["sepPercent"].iloc[0] == pytest.approx(100 * np.sqrt(np.exp(rmse ** 2) - 1))

def test_error_stats_flux_r2_exceeds_conc_r2(modeled_elist):
    # Flux varies with Q as well as C, so the same residual explains more of it.
    stats = error_stats(modeled_elist, seed=1, verbose=False)
    assert stats["RsqLogF"].iloc[0] > stats["RsqLogC"].iloc[0]

def test_error_stats_raises_without_model(base_elist):
    with pytest.raises(ValueError, match="yHat"):
        error_stats(base_elist)

def test_error_stats_does_not_change_the_elist(modeled_elist):
    before = list(modeled_elist["Sample"].columns)
    error_stats(modeled_elist, seed=1, verbose=False)
    assert list(modeled_elist["Sample"].columns) == before

def test_flux_bias_stat_columns_match_egret(modeled_elist):
    df = flux_bias_stat(modeled_elist["Sample"])
    assert list(df.columns) == ["bias1", "bias2", "bias3"]

def test_flux_bias_stat_formula(modeled_elist):
    sample = modeled_elist["Sample"]
    df = flux_bias_stat(sample)
    est = np.sum(sample["ConcHat"] * sample["Q"])
    high = np.sum(sample["ConcHigh"] * sample["Q"])
    assert df["bias1"].iloc[0] == pytest.approx((est - high) / est)

def test_flux_bias_stat_bias3_is_the_midpoint(modeled_elist):
    df = flux_bias_stat(modeled_elist["Sample"])
    # ConcLow <= ConcHigh, so the lower-bound bias is the larger one.
    assert df["bias2"].iloc[0] >= df["bias1"].iloc[0]
    midpoint = (df["bias1"].iloc[0] + df["bias2"].iloc[0]) / 2
    assert df["bias3"].iloc[0] == pytest.approx(midpoint)

def test_flux_bias_stat_raises_without_conchat(base_elist):
    with pytest.raises(ValueError, match="ConcHat"):
        flux_bias_stat(base_elist["Sample"])



# ──────────────────────────────────────────────────────────────────────────────
# table_results / table_change (R semantics)
# ──────────────────────────────────────────────────────────────────────────────

def test_table_results_columns_match_egret(modeled_elist):
    df = table_results(modeled_elist, verbose=False)
    assert list(df.columns) == ["Year", "Discharge [cms]", "Conc [mg/L]",
                                "FN Conc [mg/L]", "Flux [10^6kg/yr]",
                                "FN Flux [10^6kg/yr]"]

def test_table_results_uses_units(modeled_elist):
    annual = setup_years(modeled_elist["Daily"])
    df = table_results(modeled_elist, q_unit=1, flux_unit=3, verbose=False)
    assert df["Discharge [cfs]"].iloc[0] == pytest.approx(annual["Q"].iloc[0] * 35.314667)
    assert df["Flux [kg/day]"].iloc[0] == pytest.approx(annual["Flux"].iloc[0])

def test_table_results_year_labels(modeled_elist):
    df = table_results(modeled_elist, verbose=False)
    assert df["Year"].iloc[0] == 1991

def test_table_results_prints(modeled_elist, capsys):
    table_results(modeled_elist)
    printed = capsys.readouterr().out
    assert "Water Year" in printed
    assert "FN_Conc" in printed

def test_table_change_default_year_points(modeled_elist):
    tables = table_change(modeled_elist, verbose=False)
    conc = tables["concentration"]
    # Every fifth year back from 2001: 1991, 1996, 2001 -> three pairs.
    pairs = conc[["Year1", "Year2"]].to_numpy()
    assert np.array_equal(pairs, [[1991, 1996], [1991, 2001], [1996, 2001]])

def test_table_change_arithmetic(modeled_elist):
    annual = setup_years(modeled_elist["Daily"])
    tables = table_change(modeled_elist, year_points=[1992, 1998], verbose=False)
    first = annual["FNConc"].iloc[1]
    last = annual["FNConc"].iloc[7]
    row = tables["concentration"].iloc[0]
    assert row["change[mg/L]"] == pytest.approx(last - first)
    assert row["slope[mg/L/yr]"] == pytest.approx((last - first) / 6)
    assert row["change[%]"] == pytest.approx(100 * (last - first) / first)

def test_table_change_flux_units(modeled_elist):
    annual = setup_years(modeled_elist["Daily"])
    tables = table_change(modeled_elist, flux_unit=3, year_points=[1992, 1998], verbose=False)
    change = annual["FNFlux"].iloc[7] - annual["FNFlux"].iloc[1]
    assert tables["flux"]["change [kg/day]"].iloc[0] == pytest.approx(change)

def test_table_change_points_outside_record_fall_back(modeled_elist):
    tables = table_change(modeled_elist, year_points=[1950, 2001], verbose=False)
    assert tables["concentration"]["Year1"].iloc[0] == 1991

def test_table_change_needs_model(base_elist):
    with pytest.raises(ValueError, match="model_estimation"):
        table_change(base_elist, verbose=False)

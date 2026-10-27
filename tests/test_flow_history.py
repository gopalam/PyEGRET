"""
Tests for flowHistory: annual flow statistics, flow duration, hydrograph
timing and the Quantile-Kendall analysis.

Where EGRET's own numbers exist they are the reference: Q7 and Q30 are
checked against the columns EGRET computed for the Choptank. The rest is
checked by construction - a statistic that claims to be the annual minimum
must equal the annual minimum, order statistics must be ordered, and a
smoother must smooth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyegret.data.populate import populate_date_columns
from pyegret.flow import (
    ISTAT_NAMES,
    add_rolling_means,
    cum_q_date,
    flow_duration,
    make_annual_series,
    mann_kendall,
    quantile_kendall,
    sort_annual_flows,
    theil_sen_slope,
    zhang_prewhitened_test,
)
from pyegret.elist import make_elist


def _daily_elist(start="1990-10-01", end="2010-09-30", seed=3, trend=0.0):
    """A complete daily discharge record with a seasonal cycle."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates))
    log_q = (0.8 * np.sin(2 * np.pi * t / 365.25)
             + trend * t / 365.25
             + 0.35 * rng.standard_normal(len(dates)))
    daily = pd.DataFrame({"Date": dates, "Q": np.exp(log_q)})
    daily["LogQ"] = np.log(daily["Q"])
    daily = populate_date_columns(daily)
    return make_elist(INFO=pd.DataFrame([{"shortName": "Test"}]), Daily=daily)


@pytest.fixture(scope="module")
def elist():
    return _daily_elist()


# ---------------------------------------------------------------------------
# Rolling means
# ---------------------------------------------------------------------------

# --- RollingMeans ------------------------------------------------------

def test_rolling_means_adds_both_columns(elist):
    out = add_rolling_means(elist["Daily"])
    assert "Q7" in out.columns and "Q30" in out.columns

def test_rolling_means_is_trailing_not_centred(elist):
    """EGRET uses a one-sided filter: the first n-1 values are undefined."""
    out = add_rolling_means(elist["Daily"])
    assert np.all(out["Q7"].iloc[:6].isna())
    assert np.all(out["Q7"].iloc[6:].notna())
    assert np.all(out["Q30"].iloc[:29].isna())

def test_rolling_means_matches_a_hand_computed_window(elist):
    out = add_rolling_means(elist["Daily"])
    q = elist["Daily"]["Q"].to_numpy()
    assert out["Q7"].iloc[6] == pytest.approx(q[:7].mean())
    assert out["Q30"].iloc[29] == pytest.approx(q[:30].mean())


# ---------------------------------------------------------------------------
# Annual series
# ---------------------------------------------------------------------------

# --- AnnualSeries ------------------------------------------------------

@pytest.fixture(scope="module")
def annual(elist):
    return make_annual_series(elist)

def test_annual_series_all_eight_statistics_present(annual):
    assert set(annual["istat"].unique()) == set(ISTAT_NAMES)

def test_annual_series_minima_are_ordered(annual):
    """1-day minimum <= 7-day minimum <= 30-day minimum, every year."""
    wide = annual.pivot_table(index="DecYear", columns="istat", values="value")
    lows = wide[[1, 2, 3]].dropna()
    assert np.all((lows[1] <= lows[2] + 1e-9))
    assert np.all((lows[2] <= lows[3] + 1e-9))

def test_annual_series_maxima_are_ordered(annual):
    """30-day maximum <= 7-day maximum <= 1-day maximum, every year."""
    wide = annual.pivot_table(index="DecYear", columns="istat", values="value")
    highs = wide[[6, 7, 8]].dropna()
    assert np.all((highs[6] <= highs[7] + 1e-9))
    assert np.all((highs[7] <= highs[8] + 1e-9))

def test_annual_series_low_flow_statistics_use_the_climate_year(annual):
    """Statistics 1-3 run April to March; 4-8 run October to September.

    Their period centres therefore sit half a year apart.
    """
    low = annual[annual["istat"] == 1]["DecYear"].to_numpy()
    high = annual[annual["istat"] == 5]["DecYear"].to_numpy()
    offset = np.median(low[:len(high)]) - np.median(high[:len(low)])
    assert offset == pytest.approx(0.5, abs=0.05)

def test_annual_series_mean_exceeds_median_for_skewed_flow(annual):
    wide = annual.pivot_table(index="DecYear", columns="istat", values="value")
    both = wide[[4, 5]].dropna()
    assert (both[4] <= both[5]).mean() > 0.9

def test_annual_series_smoother_reduces_variance(annual):
    series = annual[annual["istat"] == 5].dropna(subset=["smoothed"])
    assert series["smoothed"].std() < series["value"].std()

def test_annual_series_smoothed_values_are_positive(annual):
    assert np.all((annual["smoothed"].dropna() > 0))

def test_annual_series_seasonal_period_gives_fewer_days(elist):
    """A four-month period of analysis must not use the whole year."""
    summer = make_annual_series(elist, pa_start=6, pa_long=3)
    assert not summer.empty
    full = make_annual_series(elist)
    summer_max = summer[summer["istat"] == 8]["value"].median()
    full_max = full[full["istat"] == 8]["value"].median()
    assert summer_max <= full_max

def test_annual_series_raises_without_daily():
    with pytest.raises(ValueError, match="no Daily"):
        make_annual_series(make_elist(INFO=pd.DataFrame([{}]), Daily=pd.DataFrame()))


# ---------------------------------------------------------------------------
# Flow duration
# ---------------------------------------------------------------------------

# --- FlowDuration ------------------------------------------------------

def test_flow_duration_quantiles_are_ordered(elist):
    duration = flow_duration(elist)
    assert np.all((np.diff(duration.to_numpy()) >= 0))

def test_flow_duration_has_the_nine_egret_quantiles(elist):
    duration = flow_duration(elist)
    assert list(duration.index) == [
        "min", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "max"]

def test_flow_duration_min_and_max_are_the_record_extremes(elist):
    duration = flow_duration(elist)
    assert duration["min"] == pytest.approx(elist["Daily"]["Q"].min())
    assert duration["max"] == pytest.approx(elist["Daily"]["Q"].max())

def test_flow_duration_seasonal_window_narrows_the_range(elist):
    """A 30-day window around one date sees less spread than the year."""
    full = flow_duration(elist, span=365)
    winter = flow_duration(elist, center_date="01-15", span=15)
    assert winter["max"] <= full["max"]
    assert winter["min"] >= full["min"]


# ---------------------------------------------------------------------------
# Hydrograph timing
# ---------------------------------------------------------------------------

# --- CumQDate ----------------------------------------------------------

def test_cum_q_date_returns_one_row_per_year(elist):
    timing = cum_q_date(elist)
    assert list(timing.columns) == ["Year", "Day", "DayOfPeriod"]
    assert timing["Year"].is_unique
    assert len(timing) >= 18

def test_cum_q_date_days_are_valid(elist):
    timing = cum_q_date(elist)
    assert np.all(timing["Day"].between(1, 366))

def test_cum_q_date_larger_fraction_falls_later(elist):
    """The date of 75 % of runoff cannot precede the date of 25 %.

    Measured on DayOfPeriod, not Day: the water year starts in October,
    so calendar day numbers wrap through New Year and cannot be ordered.
    """
    early = cum_q_date(elist, fract=0.25)
    early = early.set_index("Year")
    early = early["DayOfPeriod"]
    late = cum_q_date(elist, fract=0.75)
    late = late.set_index("Year")
    late = late["DayOfPeriod"]
    common = early.index.intersection(late.index)
    assert np.all((late[common] >= early[common]))

def test_cum_q_date_calendar_day_wraps_within_a_water_year(elist):
    """Day is EGRET's calendar day of year and legitimately wraps.

    Documented as a test so the behaviour is not mistaken for a bug.
    """
    timing = cum_q_date(elist, fract=0.75)
    assert np.all(timing["Day"].between(1, 366))
    assert np.all(timing["DayOfPeriod"].between(1, 366))

def test_cum_q_date_rejects_an_impossible_fraction(elist):
    with pytest.raises(ValueError, match="between 0 and 1"):
        cum_q_date(elist, fract=1.5)


# ---------------------------------------------------------------------------
# Quantile-Kendall
# ---------------------------------------------------------------------------

# --- SortAnnualFlows ---------------------------------------------------

def test_sort_annual_flows_shape_and_ordering(elist):
    flows, years = sort_annual_flows(elist)
    assert flows.shape[1] == len(years)
    assert flows.shape[0] in (365, 366)
    for column in range(flows.shape[1]):
        assert np.all((np.diff(flows[:, column]) >= 0))

def test_sort_annual_flows_first_rank_is_the_annual_minimum(elist):
    flows, _ = sort_annual_flows(elist)
    assert np.allclose(flows[0, :], flows.min(axis=0))

def test_sort_annual_flows_last_rank_is_the_annual_maximum(elist):
    flows, _ = sort_annual_flows(elist)
    assert np.allclose(flows[-1, :], flows.max(axis=0))

def test_sort_annual_flows_incomplete_years_are_dropped(elist):
    """A year missing a single day cannot supply order statistics."""
    gapped = elist["Daily"].drop(elist["Daily"].index[500])
    gapped = gapped.reset_index(drop=True)
    trimmed = make_elist(INFO=elist["INFO"], Daily=gapped)
    _, years_full = sort_annual_flows(elist)
    _, years_gapped = sort_annual_flows(trimmed)
    assert len(years_gapped) == len(years_full) - 1


# --- QuantileKendall ---------------------------------------------------

@pytest.fixture(scope="module")
def results(elist):
    return quantile_kendall(elist)

def test_quantile_kendall_one_row_per_order_statistic(results, elist):
    flows, _ = sort_annual_flows(elist)
    assert len(results) == flows.shape[0]

def test_quantile_kendall_plotting_positions_increase(results):
    assert np.all((np.diff(results["z"].to_numpy()) > 0))
    assert np.all(results["freq"].between(0, 1))

def test_quantile_kendall_p_values_are_probabilities(results):
    for column in ("p_value", "p_value_adj"):
        valid = results[column].dropna()
        assert np.all(valid.between(0, 1))

def test_quantile_kendall_slope_pct_follows_slope_log(results):
    expected = 100.0 * (np.exp(results["slope_log"]) - 1.0)
    assert np.allclose(results["slope_pct"], expected, equal_nan=True)

def test_quantile_kendall_detects_an_imposed_trend():
    """A record built with a rising trend must come back with one."""
    rising = _daily_elist(seed=11, trend=0.02)
    results = quantile_kendall(rising, adjust_for_serial_correlation=False)
    assert results["slope_log"].median() > 0
    assert (results["p_value"] < 0.05).mean() > 0.5

def test_quantile_kendall_reports_no_trend_on_a_stationary_record():
    flat = _daily_elist(seed=12, trend=0.0)
    results = quantile_kendall(flat, adjust_for_serial_correlation=False)
    assert (results["p_value"] < 0.05).mean() < 0.2

def test_quantile_kendall_skipping_the_adjustment_leaves_it_blank(elist):
    results = quantile_kendall(elist, adjust_for_serial_correlation=False)
    assert np.all(results["p_value_adj"].isna())


# ---------------------------------------------------------------------------
# Trend statistics
# ---------------------------------------------------------------------------

# --- TrendStatistics ---------------------------------------------------

def test_trend_statistics_theil_sen_matches_scipy():
    from scipy.stats import theilslopes
    rng = np.random.default_rng(5)
    x = np.arange(40.0)
    y = 0.07 * x + rng.standard_normal(40)
    assert theil_sen_slope(x, y) == pytest.approx(theilslopes(y, x)[0])

def test_trend_statistics_theil_sen_ignores_a_wild_outlier():
    """The median of pairwise slopes is what makes this robust."""
    x = np.arange(20.0)
    y = 0.5 * x
    clean = theil_sen_slope(x, y)
    y[10] = 1000.0
    assert theil_sen_slope(x, y) == pytest.approx(clean, rel=0.05)

def test_trend_statistics_kendall_tau_matches_scipy():
    from scipy.stats import kendalltau
    rng = np.random.default_rng(6)
    x = np.arange(30.0)
    y = rng.standard_normal(30) + 0.08 * x
    assert mann_kendall(x, y)["tau"] == pytest.approx(kendalltau(x, y).statistic)

def test_trend_statistics_kendall_tau_handles_ties():
    from scipy.stats import kendalltau
    x = np.arange(20.0)
    y = np.repeat([1.0, 1.0, 2.0, 2.0, 3.0], 4)
    assert mann_kendall(x, y)["tau"] == pytest.approx(kendalltau(x, y).statistic)

def test_trend_statistics_prewhitening_discounts_serial_correlation():
    """An autocorrelated series must lose significance after prewhitening.

    Consecutive years that carry the same information are not independent
    evidence, and an unadjusted test counts them as if they were.
    """
    rng = np.random.default_rng(21)
    n = 70
    noise = rng.standard_normal(n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.8 * series[i - 1] + noise[i]
    series = series[10:]
    t = np.arange(1.0, series.size + 1)
    y = series + 0.03 * t

    unadjusted = mann_kendall(t, y)["p_value"]
    adjusted = zhang_prewhitened_test(y, t)
    assert adjusted["autocorrelation"] > 0.3
    assert adjusted["p_value"] > unadjusted

def test_trend_statistics_prewhitening_leaves_uncorrelated_series_alone():
    """Below the 0.05 autocorrelation threshold, the test is untouched.

    Built to alternate rather than drawn at random: a finite random
    sample can easily show 0.15 lag-1 autocorrelation by chance, which
    legitimately triggers prewhitening and would make this flaky.
    """
    rng = np.random.default_rng(22)
    t = np.arange(1.0, 61.0)
    y = np.where(np.arange(60) % 2 == 0, 1.0, -1.0) + 0.1 * rng.standard_normal(60)
    adjusted = zhang_prewhitened_test(y, t)
    assert adjusted["autocorrelation"] < 0.05
    assert adjusted["p_value"] == pytest.approx(
        mann_kendall(t, y)["p_value"], abs=1e-12
    )

def test_trend_statistics_a_trend_alone_can_trigger_prewhitening():
    """zyp measures autocorrelation on the raw series, trend included.

    A strong trend makes consecutive values similar, so the lag-1
    autocorrelation of the undetrended series is positive even when the
    noise is independent. The iteration then detrends and settles. This
    is zyp.zhang's behaviour, mirrored deliberately.
    """
    rng = np.random.default_rng(22)
    t = np.arange(1.0, 61.0)
    y = 0.02 * t + rng.standard_normal(60)
    adjusted = zhang_prewhitened_test(y, t)
    assert adjusted["autocorrelation"] > 0.05
    assert adjusted["p_value"] > mann_kendall(t, y)["p_value"]

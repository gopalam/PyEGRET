"""
Tests for the wall and for generalized flow normalization.

Both exist to relax an assumption WRTDS makes by default, so both are tested
the same way: build a record that violates the assumption, and check that the
default gets it wrong and the correction gets it right. A test that only
showed the correction running would not distinguish it from doing nothing.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pyegret.analysis.run_pairs import run_pairs
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import est_surfaces
from pyegret.model.flex_fn import flex_fn, make_date_info
from pyegret.model.wall import decimal_high_low, stitch_surfaces
from pyegret.testing import annual_mean_by_calendar_year, make_truth, make_synthetic_elist, truth_conc

WALL_DATE = "2000-01-01"
STEP = -0.40          # a 33 % drop in concentration


def _annual_fn(daily, column="FNConc"):
    return annual_mean_by_calendar_year(daily, column)


def _true_annual_fn(elist, step=0.0, step_date=None):
    """Flow-normalized truth: the true surface integrated over each day-of-year's flows."""
    daily = elist["Daily"]
    doy = daily["Day"].to_numpy()
    log_q = daily["LogQ"].to_numpy()
    dec_year = daily["DecYear"].to_numpy()
    by_doy = {k: log_q[doy == k] for k in np.unique(doy)}

    shift = np.zeros(len(daily))
    if step:
        dates = pd.to_datetime(daily["Date"])
        after = dates.to_numpy() > np.datetime64(pd.Timestamp(step_date))
        shift = step * after

    values = np.array([
        np.mean(truth_conc(np.full(by_doy[doy[k]].size, dec_year[k]), by_doy[doy[k]]))
        * np.exp(shift[k])
        for k in range(len(daily))
    ])
    frame = pd.DataFrame({"DecYear": daily["DecYear"].to_numpy(), "value": values})
    return annual_mean_by_calendar_year(frame, "value")


# ---------------------------------------------------------------------------
# The wall
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stepped():
    """A record with a known abrupt drop part way through."""
    return make_synthetic_elist(step_change=STEP, step_date=WALL_DATE, seed=5)


@pytest.fixture(scope="module")
def without_wall(stepped):
    elist = copy.deepcopy(stepped)
    elist["surfaces"] = est_surfaces(elist, verbose=False)
    elist["Daily"] = est_daily_from_surfaces(elist)
    return elist


@pytest.fixture(scope="module")
def with_wall(stepped):
    elist = copy.deepcopy(stepped)
    elist["surfaces"] = stitch_surfaces(elist, wall_date=WALL_DATE, verbose=False)
    elist["Daily"] = est_daily_from_surfaces(elist)
    return elist


# --- DecimalHighLow ----------------------------------------------------
# Edge adjustment measures distance to the water-year bounds, not the samples.

def test_decimal_high_low_october_start_rounds_up_to_the_water_year():
    sample = pd.DataFrame({"DecYear": [1979.81, 2011.74], "Month": [10, 9]})
    low, high = decimal_high_low(sample)
    assert low == pytest.approx(1979.75)
    assert high == pytest.approx(2011.75)

def test_decimal_high_low_january_start_stays_in_the_calendar_year():
    sample = pd.DataFrame({"DecYear": [1980.05, 2011.10], "Month": [1, 2]})
    low, high = decimal_high_low(sample)
    assert low == pytest.approx(1979.75)
    assert high == pytest.approx(2011.75)


# --- WallMechanics -----------------------------------------------------

def test_wall_mechanics_surface_has_the_standard_shape(with_wall):
    assert with_wall["surfaces"].ndim == 3
    assert with_wall["surfaces"].shape[1] == len(with_wall["vector_logQ"])
    assert with_wall["surfaces"].shape[0] == len(with_wall["vector_year"])

def test_wall_mechanics_conc_hat_is_rebuilt_from_the_averaged_parts(with_wall):
    """exp() does not commute with a mean, so the blend recomputes it."""
    surfaces = with_wall["surfaces"]
    finite = np.isfinite(surfaces[:, :, 0]) & np.isfinite(surfaces[:, :, 1])
    expected = np.exp(surfaces[finite][:, 0] + 0.5 * surfaces[finite][:, 1] ** 2)
    assert np.allclose(surfaces[finite][:, 2], expected, rtol=1e-9)

def test_wall_mechanics_produces_usable_daily_estimates(with_wall):
    assert np.mean(np.isfinite(with_wall["Daily"]["FNConc"].to_numpy())) > 0.99

def test_wall_mechanics_rejects_a_wall_outside_the_samples(stepped):
    with pytest.raises(ValueError, match="both sides need data"):
        stitch_surfaces(copy.deepcopy(stepped), wall_date="1899-01-01",
                        verbose=False)

def test_wall_mechanics_relaxes_the_observation_minimum(stepped, capsys):
    """Each half holds fewer samples, so the minimum has to come down."""
    stitch_surfaces(copy.deepcopy(stepped), wall_date=WALL_DATE, verbose=True)
    output = capsys.readouterr().out
    assert "relaxed to" in output


# --- WallRecoversTheStep -----------------------------------------------

def _step_size(elist):
    annual = _annual_fn(elist["Daily"])
    before = annual[[1997, 1998, 1999]].mean()
    after = annual[[2000, 2001, 2002]].mean()
    return float(np.log(after / before))

def test_wall_recovers_the_step_smoothing_across_the_wall_erases_the_step(without_wall, stepped):
    """The default assumption of gradual change hides an abrupt one."""
    truth = _true_annual_fn(stepped, step=STEP, step_date=WALL_DATE)
    true_step = float(np.log(truth[[2000, 2001, 2002]].mean()
                             / truth[[1997, 1998, 1999]].mean()))
    recovered = _step_size(without_wall) / true_step
    assert recovered < 0.25

def test_wall_recovers_the_step_the_wall_recovers_most_of_the_step(with_wall, stepped):
    truth = _true_annual_fn(stepped, step=STEP, step_date=WALL_DATE)
    true_step = float(np.log(truth[[2000, 2001, 2002]].mean()
                             / truth[[1997, 1998, 1999]].mean()))
    recovered = _step_size(with_wall) / true_step
    # Overshoots somewhat: each half is short and the fit at the boundary
    # extrapolates from one side only. Direction and scale are right.
    assert 0.7 < recovered < 1.8

def test_wall_recovers_the_step_the_wall_reduces_the_error(with_wall, without_wall, stepped):
    truth = _true_annual_fn(stepped, step=STEP, step_date=WALL_DATE)

    def error(elist):
        annual = _annual_fn(elist["Daily"])
        common = annual.index.intersection(truth.index)
        return float(np.nanmedian(
            np.abs((annual[common] - truth[common]) / truth[common])
        ))

    assert error(with_wall) < error(without_wall) / 1.5


# ---------------------------------------------------------------------------
# Generalized flow normalization
# ---------------------------------------------------------------------------

# --- MakeDateInfo ------------------------------------------------------

@pytest.fixture(scope="module")
def elist():
    return make_synthetic_elist(seed=17)

def test_make_date_info_stationary_gives_one_segment(elist):
    info = make_date_info(elist, window_side=0)
    assert len(info) == 1
    assert info["flowNormStart"].iloc[0] == pd.to_datetime(elist["Daily"]["Date"]).iloc[0]
    assert info["flowNormEnd"].iloc[0] == pd.to_datetime(elist["Daily"]["Date"]).iloc[-1]

def test_make_date_info_sliding_window_gives_one_segment_per_year(elist):
    info = make_date_info(elist, window_side=5)
    assert len(info) >= 18
    assert np.all((info["flowStart"] < info["flowEnd"]))

def test_make_date_info_window_widens_the_reference_period(elist):
    info = make_date_info(elist, window_side=5)
    middle = info.iloc[len(info) // 2]
    span_years = (middle["flowNormEnd"] - middle["flowNormStart"]).days / 365.25
    assert span_years > 5

def test_make_date_info_flow_break_splits_the_record(elist):
    info = make_date_info(elist, window_side=0, flow_break_date="2000-06-30")
    assert len(info) == 2
    assert info["flowNormEnd"].iloc[0] == pd.Timestamp("2000-06-30")
    assert info["flowNormStart"].iloc[1] == pd.Timestamp("2000-07-01")

def test_make_date_info_flow_break_never_mixes_the_two_eras(elist):
    """The point of a break is that flows either side are not comparable."""
    info = make_date_info(elist, window_side=4, flow_break_date="2000-06-30")
    break_date = pd.Timestamp("2000-06-30")
    before = info[info["flowEnd"] <= break_date]
    after = info[info["flowStart"] > break_date]
    assert np.all((before["flowNormEnd"] <= break_date))
    assert np.all((after["flowNormStart"] > break_date))

def test_make_date_info_rejects_a_break_outside_the_record(elist):
    with pytest.raises(ValueError, match="outside the discharge record"):
        make_date_info(elist, flow_break_date="1899-01-01")


# --- FlexFn ------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    elist = make_synthetic_elist(seed=19)
    elist["surfaces"] = est_surfaces(elist, verbose=False)
    return elist

def test_flex_fn_stationary_matches_plain_flow_normalization(fitted):
    """window_side=0 is standard flow normalization, the special case."""
    flexible = flex_fn(fitted, window_side=0)
    plain = est_daily_from_surfaces(fitted)
    assert np.allclose(flexible["FNConc"], plain["FNConc"], equal_nan=True)

def test_flex_fn_every_day_is_covered(fitted):
    result = flex_fn(fitted, window_side=5)
    assert np.all(result["FNConc"].notna())

def test_flex_fn_moving_window_differs_from_stationary(fitted):
    moving = flex_fn(fitted, window_side=3)
    plain = est_daily_from_surfaces(fitted)
    assert not np.allclose(moving["FNConc"], plain["FNConc"], equal_nan=True)

def test_flex_fn_conc_day_does_not_depend_on_normalization(fitted):
    """Only the flow-normalized pair should move."""
    moving = flex_fn(fitted, window_side=5)
    plain = est_daily_from_surfaces(fitted)
    assert np.allclose(moving["ConcDay"], plain["ConcDay"], equal_nan=True)

def test_flex_fn_requires_a_fitted_surface():
    with pytest.raises(ValueError, match="surfaces is None"):
        flex_fn(make_synthetic_elist(seed=1))


# --- TrendDecomposition ------------------------------------------------
# The reason GFN exists: separating a wetter watershed from a leakier one.

def _fit(**kwargs):
    elist = make_synthetic_elist(**kwargs)
    elist["surfaces"] = est_surfaces(elist, verbose=False)
    return elist

@pytest.fixture(scope="module")
def flows_changed():
    """Discharge trends upward; the concentration-discharge relation is fixed."""
    return _fit(truth=make_truth(trend=0.0), flow_trend=0.03, seed=31)

@pytest.fixture(scope="module")
def watershed_changed():
    """The relation shifts; the flow distribution is stationary."""
    return _fit(truth=make_truth(trend=-0.03), flow_trend=0.0, seed=31)

def _pairs(elist, window_side):
    return run_pairs(elist, 1992, 2008, window_side=window_side, verbose=False)

def test_trend_decomposition_flow_driven_change_lands_in_qtc(flows_changed):
    results = _pairs(flows_changed, 6)
    row = results["pairResults"].loc["Conc"]
    assert results["Other"]["PercentChangeConc"]["Total Percent Change"] > 5
    assert abs(row["QTC"]) > abs(row["CQTC"])

def test_trend_decomposition_watershed_driven_change_lands_in_cqtc(watershed_changed):
    results = _pairs(watershed_changed, 6)
    row = results["pairResults"].loc["Conc"]
    assert results["Other"]["PercentChangeConc"]["Total Percent Change"] < -5
    assert abs(row["CQTC"]) > 5 * abs(row["QTC"])

def test_trend_decomposition_components_sum_to_the_total(flows_changed):
    results = _pairs(flows_changed, 6)
    for label in ("Conc", "Flux"):
        row = results["pairResults"].loc[label]
        assert row["CQTC"] + row["QTC"] == pytest.approx(row["TotalChange"])

def test_trend_decomposition_total_change_is_x22_minus_x11(flows_changed):
    row = _pairs(flows_changed, 6)["pairResults"].loc["Conc"]
    assert row["TotalChange"] == pytest.approx(row["x22"] - row["x11"])

def test_trend_decomposition_cqtc_is_x20_minus_x10(flows_changed):
    row = _pairs(flows_changed, 6)["pairResults"].loc["Conc"]
    assert row["CQTC"] == pytest.approx(row["x20"] - row["x10"])

def test_trend_decomposition_stationary_normalization_cannot_attribute_anything_to_flow(flows_changed):
    """With one fixed reference distribution, QTC is zero by construction."""
    results = _pairs(flows_changed, 0)
    assert results["pairResults"].loc["Conc", "QTC"] == pytest.approx(0.0, abs=1e-9)

def test_trend_decomposition_stationary_normalization_understates_a_flow_driven_change(flows_changed):
    moving = _pairs(flows_changed, 6)["Other"]["PercentChangeConc"]["Total Percent Change"]
    fixed = _pairs(flows_changed, 0)["Other"]["PercentChangeConc"]["Total Percent Change"]
    assert moving > fixed + 5

def test_trend_decomposition_flux_is_reported_in_million_kg_per_year(flows_changed):
    results = _pairs(flows_changed, 6)
    assert 0 < abs(results["pairResults"].loc["Flux", "x11"]) < 1000

def test_trend_decomposition_date_info_has_one_row_per_year(flows_changed):
    info = _pairs(flows_changed, 6)["dateInfo"]
    assert len(info) == 2
    assert info["flowStart"].iloc[0] == pd.Timestamp("1991-10-01")
    assert info["flowEnd"].iloc[1] == pd.Timestamp("2008-09-30")

def test_trend_decomposition_fits_its_own_surfaces():
    # Like R runPairs, no fitted surface is needed: it fits one per year.
    elist = make_synthetic_elist(seed=1)
    results = run_pairs(elist, 1992, 2008, window_side=0, verbose=False)
    assert np.isfinite(results["pairResults"].loc["Conc", "x11"])

def test_trend_decomposition_old_surface_agrees_with_refit(flows_changed):
    # The full-record surface and a per-year refit use the same samples and
    # windows, so they should agree closely.
    refit = _pairs(flows_changed, 0)["pairResults"].loc["Conc", "x11"]
    old = run_pairs(flows_changed, 1992, 2008, window_side=0, old_surface=True,
                    verbose=False)["pairResults"].loc["Conc", "x11"]
    assert old == pytest.approx(refit, rel=0.02)

def test_trend_decomposition_year_outside_samples_is_an_error(flows_changed):
    with pytest.raises(ValueError, match="year2 is outside"):
        run_pairs(flows_changed, 1992, 2030, window_side=0, verbose=False)

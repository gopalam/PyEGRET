"""
Does the WRTDS engine recover a known answer?

Every other test in this suite checks plumbing: shapes, column names, that
values come back finite. None of them would notice the engine reporting a
declining trend as a rising one, which is exactly what a coarse surface grid
and a mis-specified seasonal kernel caused it to do.

These tests run the full pipeline over a synthetic record whose true
concentration surface is prescribed (:mod:`pyegret.testing`) and check the
answers against it. They are slower than the rest of the suite - a few
seconds - and they are the ones that matter.

Tolerances are set with generous headroom over the errors actually observed,
so ordinary numerical drift will not trip them; only a real regression will.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyegret.model.model_estimation import model_estimation
from pyegret.testing import (
    DEFAULT_TRUTH,
    annual_mean_by_calendar_year,
    make_synthetic_elist,
    truth_change_over,
    truth_conc,
)

_TWO_PI = 2.0 * np.pi


def _seasonal_amplitude(values: np.ndarray, dec_year: np.ndarray) -> float:
    """Amplitude of the annual cycle in log(values), by least squares."""
    design = np.column_stack([
        np.ones_like(dec_year),
        np.sin(_TWO_PI * dec_year),
        np.cos(_TWO_PI * dec_year),
    ])
    beta = np.linalg.lstsq(design, np.log(values), rcond=None)[0]
    return float(np.hypot(beta[1], beta[2]))


def _annual_fn_conc(daily):
    """Annual mean flow-normalized concentration, and the years."""
    annual = annual_mean_by_calendar_year(daily, "FNConc")
    return annual.index.to_numpy(dtype=float), annual.to_numpy(dtype=float)


@pytest.fixture(scope="module")
def truth_run():
    """Fit the full WRTDS pipeline to a synthetic record with a known truth."""
    elist = make_synthetic_elist()
    return model_estimation(elist, verbose=False)


# --- SurfaceQuality ----------------------------------------------------

def test_surface_quality_grid_resolves_the_seasonal_cycle(truth_run):
    """The year axis must be finer than the signal it has to represent.

    Season enters the regression through SinDY/CosDY evaluated at the
    grid year, so a step coarser than a season cannot carry it.
    """
    step = float(np.diff(truth_run["vector_year"])[0])
    assert step <= 1.0 / 12.0

def test_surface_quality_no_unestimated_grid_points(truth_run):
    """Window expansion should leave no NaN cells to interpolate through."""
    assert np.all(np.isfinite(truth_run["surfaces"][:, :, 0]))

def test_surface_quality_logq_axis_covers_the_daily_record(truth_run):
    """The surface is interpolated onto every day, including flow extremes."""
    v_logQ = truth_run["vector_logQ"]
    assert v_logQ[0] <= truth_run["Daily"]["LogQ"].min()
    assert v_logQ[-1] >= truth_run["Daily"]["LogQ"].max()


# --- DailyEstimates ----------------------------------------------------

def test_daily_estimates_concday_is_unbiased(truth_run):
    daily = truth_run["Daily"]
    expected = truth_conc(daily["DecYear"].to_numpy(), daily["LogQ"].to_numpy())
    rel = (daily["ConcDay"].to_numpy() - expected) / expected
    assert abs(float(np.nanmedian(rel))) < 0.05

def test_daily_estimates_concday_spread_is_tight(truth_run):
    """Interquartile spread of relative error, well inside the +-28/+37 %
    the coarse-grid implementation produced."""
    daily = truth_run["Daily"]
    expected = truth_conc(daily["DecYear"].to_numpy(), daily["LogQ"].to_numpy())
    rel = (daily["ConcDay"].to_numpy() - expected) / expected
    iqr = float(np.nanpercentile(rel, 75) - np.nanpercentile(rel, 25))
    assert iqr < 0.15

def test_daily_estimates_seasonality_is_recovered(truth_run):
    """Model and truth must show the same annual cycle.

    Compared against the truth evaluated at the same discharges rather
    than against the surface coefficient, because observed concentration
    also swings seasonally through its dependence on discharge.
    """
    daily = truth_run["Daily"]
    dec_year = daily["DecYear"].to_numpy()
    expected = truth_conc(dec_year, daily["LogQ"].to_numpy())
    modelled = _seasonal_amplitude(daily["ConcDay"].to_numpy(), dec_year)
    actual = _seasonal_amplitude(expected, dec_year)
    assert modelled == pytest.approx(actual, abs=0.10)


# --- FlowNormalizedTrend -----------------------------------------------
# The headline WRTDS output.

def test_flow_normalized_trend_trend_direction_is_correct(truth_run):
    """The truth declines; the estimate must decline too.

    This is the assertion that would have caught the sign reversal.
    """
    years, fn_conc = _annual_fn_conc(truth_run["Daily"])
    assert fn_conc[-1] < fn_conc[0]

def test_flow_normalized_trend_trend_magnitude(truth_run):
    years, fn_conc = _annual_fn_conc(truth_run["Daily"])
    slope = float(np.polyfit(years, np.log(fn_conc), 1)[0])
    assert slope == pytest.approx(DEFAULT_TRUTH["trend"], abs=0.005)

def test_flow_normalized_trend_change_over_record(truth_run):
    years, fn_conc = _annual_fn_conc(truth_run["Daily"])
    observed = float(fn_conc[-1] / fn_conc[0] - 1.0)
    expected = truth_change_over(DEFAULT_TRUTH, years[-1] - years[0])
    assert observed == pytest.approx(expected, abs=0.05)

def test_flow_normalized_trend_flow_normalization_removes_flow_variability(truth_run):
    """FNConc should vary far less year to year than the raw estimate."""
    daily = truth_run["Daily"]
    fn = np.asarray(annual_mean_by_calendar_year(daily, "FNConc"))
    raw = np.asarray(annual_mean_by_calendar_year(daily, "ConcDay"))
    fn_cv = np.std(fn, ddof=1) / np.mean(fn)
    raw_cv = np.std(raw, ddof=1) / np.mean(raw)
    assert fn_cv < raw_cv


# --- FluxFormulas ------------------------------------------------------

def test_flux_formulas_fluxday_is_concday_times_q(truth_run):
    daily = truth_run["Daily"]
    expected = daily["ConcDay"] * daily["Q"] * 86.4
    assert np.allclose(daily["FluxDay"], expected, rtol=1e-9, equal_nan=True)

def test_flux_formulas_fnflux_averages_the_product(truth_run):
    """FNFlux is mean(conc x Q) over reference flows, not FNConc x Q.

    Concentration rises with discharge in the truth model, so averaging
    the product must exceed the product of the averages.
    """
    daily = truth_run["Daily"]
    naive = (daily["FNConc"] * daily["Q"] * 86.4).to_numpy()
    actual = daily["FNFlux"].to_numpy()
    assert np.nanmedian(actual / naive) > 1.05


# --- BiasCorrection ----------------------------------------------------

def test_bias_correction_concday_carries_the_lognormal_correction(truth_run):
    """ConcDay must come from the ConcHat surface, exp(yHat + SE^2/2).

    Using exp(yHat) instead biases every concentration and flux low by
    exp(SE^2/2).
    """
    surfaces = truth_run["surfaces"]
    correction = np.nanmean(np.exp(0.5 * surfaces[:, :, 1] ** 2))
    assert correction > 1.0        # the correction is real, not a no-op

    conc_hat = surfaces[:, :, 2]
    y_hat = surfaces[:, :, 0]
    se = surfaces[:, :, 1]
    assert np.allclose(conc_hat, np.exp(y_hat + 0.5 * se ** 2), rtol=1e-9)


# --- CensoredData ------------------------------------------------------

def test_censored_data_handles_censored_samples():
    """A record with non-detects must still recover the trend direction."""
    elist = make_synthetic_elist(censor_fraction=0.25, seed=7)
    fitted = model_estimation(elist, verbose=False)
    years, fn_conc = _annual_fn_conc(fitted["Daily"])
    assert np.all(np.isfinite(fn_conc))
    assert fn_conc[-1] < fn_conc[0]

"""
Phase 3 tests: tricube kernel, censored regression, surface estimation,
daily interpolation, and model_estimation pipeline.

Synthetic datasets are kept deliberately small so the suite completes in
a few seconds even on slow machines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyegret.model.tri_cube import tri_cube, compute_weights, season_distance
from pyegret.model.run_surv_reg import run_surv_reg
from pyegret.model.est_surfaces import est_surfaces
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.model_estimation import model_estimation
from pyegret.elist import make_elist, has_model
from pyegret.data.populate import populate_date_columns, populate_sample_columns

_RNG = np.random.default_rng(42)
_TWO_PI = 2.0 * np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic eList fixture (small grid: 4 year × 3 logQ)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_elist(n_sample: int = 130, n_daily: int = 730) -> dict:
    """Build a minimal but realistic eList for integration tests."""
    # Daily
    dates_d = pd.date_range("2000-01-01", periods=n_daily, freq="D")
    q_d = _RNG.uniform(0.5, 50.0, n_daily)
    daily = pd.DataFrame({"Date": dates_d, "Q": q_d})
    daily["LogQ"] = np.log(daily["Q"])
    daily = populate_date_columns(daily)

    # Sample: 130 observations scattered through the record, enough for R's
    # default minimums (100 samples, 50 uncensored), which runSurvReg enforces.
    rng_idx = _RNG.integers(0, n_daily, size=n_sample)
    sample_dates = dates_d[rng_idx]
    conc = _RNG.lognormal(mean=0.2, sigma=0.4, size=n_sample)
    uncen = (_RNG.uniform(size=n_sample) > 0.15).astype(int)

    sample = pd.DataFrame({
        "Date": sample_dates,
        "ConcHigh": conc,
        "ConcLow": np.where(uncen == 0, 0.0, conc),
        "Uncen": uncen,
    })
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)

    # Merge Q onto Sample
    daily_q = daily[["Date", "Q", "LogQ"]].sort_values("Date")
    sample = sample.sort_values("Date")
    sample = pd.merge_asof(
        sample, daily_q, on="Date", direction="nearest",
        tolerance=pd.Timedelta("1 day")
    )
    sample = sample.reset_index(drop=True)

    info = pd.DataFrame({
        "station_nm": ["Synthetic River"],
        "site_no":    ["00000000"],
        "shortName":  ["Synthetic River"],
        "paramShortName": ["Test Constituent"],
        "param_cd":   ["99999"],
        "lat_va": [39.0], "long_va": [-76.0], "drain_area_va": [100.0],
    })

    return make_elist(INFO=info, Daily=daily, Sample=sample)


@pytest.fixture(scope="module")
def synthetic_elist():
    return _make_synthetic_elist()


@pytest.fixture(scope="module")
def modeled_elist(synthetic_elist):
    """Run model_estimation on the synthetic eList once per module."""
    import copy
    el = copy.deepcopy(synthetic_elist)
    return model_estimation(el, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# tri_cube
# ─────────────────────────────────────────────────────────────────────────────

# --- TriCube -----------------------------------------------------------

def test_tri_cube_zero_outside_window():
    x = np.array([-1.5, -1.0, 1.0, 1.5])
    w = tri_cube(x, 1.0)
    assert w[0] == 0.0
    assert w[3] == 0.0

def test_tri_cube_zero_at_boundary():
    # |x| == h → u == 1 → (1-1)^3 = 0
    assert tri_cube(np.array([1.0]), 1.0)[0] == 0.0
    assert tri_cube(np.array([-1.0]), 1.0)[0] == 0.0

def test_tri_cube_one_at_center():
    assert tri_cube(np.array([0.0]), 1.0)[0] == pytest.approx(1.0)

def test_tri_cube_all_nonneg():
    x = np.linspace(-2.0, 2.0, 200)
    assert np.all((tri_cube(x, 1.0) >= 0))

def test_tri_cube_symmetry():
    x = np.linspace(-0.9, 0.9, 19)
    w = tri_cube(x, 1.0)
    assert np.allclose(w, w[::-1])

def test_tri_cube_2d_array():
    x = np.array([[-0.5, 0.0], [0.5, 1.5]])
    w = tri_cube(x, 1.0)
    assert w.shape == (2, 2)
    assert w[1, 1] == 0.0   # 1.5 outside window

def test_tri_cube_scalar_input():
    assert tri_cube(np.array([0.0]), 2.0)[0] == pytest.approx(1.0)

def test_tri_cube_known_value():
    # u = 0.5 → (1 - 0.125)³ = 0.875³ ≈ 0.6699
    w = tri_cube(np.array([0.5]), 1.0)[0]
    assert w == pytest.approx((1 - 0.5**3)**3, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# compute_weights
# ─────────────────────────────────────────────────────────────────────────────

# --- ComputeWeights ----------------------------------------------------

@pytest.fixture
def sample_arrays():
    n = 20
    years = np.linspace(1990, 2010, n)
    logq = np.linspace(0, 4, n)
    return years, logq

def test_compute_weights_zero_for_far_target(sample_arrays):
    years, logq = sample_arrays
    w = compute_weights(
        target_year=2050.0, target_logQ=0.0,
        sample_years=years, sample_logQ=logq,
    )
    assert w.sum() < 1e-6

def test_compute_weights_all_nonneg(sample_arrays):
    years, logq = sample_arrays
    w = compute_weights(2000.0, 2.0, years, logq)
    assert np.all((w >= 0))

def test_compute_weights_output_shape(sample_arrays):
    years, logq = sample_arrays
    w = compute_weights(2000.0, 2.0, years, logq)
    assert w.shape == (20,)

def test_compute_weights_nearby_sample_gets_high_weight():
    # One sample exactly at the target -> maximum weight
    w = compute_weights(2000.0, 2.0, np.array([2000.0]), np.array([2.0]))
    assert w[0] == pytest.approx(1.0)

def test_compute_weights_product_of_three_components():
    """Weight must equal w_time x w_logQ x w_season."""
    yr, lq = 2000.0, 2.0
    sy = np.array([1998.25])
    sq = np.array([1.5])

    w = compute_weights(yr, lq, sy, sq)
    wt = tri_cube(sy - yr, 7.0)
    wq = tri_cube(sq - lq, 2.0)
    ws = tri_cube(season_distance(sy, yr), 0.5)
    assert w[0] == pytest.approx((wt * wq * ws)[0], rel=1e-9)


# --- SeasonDistance ----------------------------------------------------
# EGRET's seasonal distance: min(dt mod 1, 1 - dt mod 1), in [0, 0.5].

def test_season_distance_same_day_of_year_is_zero():
    # Ten years apart, same point in the season
    d = season_distance(np.array([2010.3]), 2000.3)
    assert d[0] == pytest.approx(0.0, abs=1e-9)

def test_season_distance_opposite_season_is_half():
    d = season_distance(np.array([2000.5]), 2000.0)
    assert d[0] == pytest.approx(0.5)

def test_season_distance_bounded_to_half_year():
    d = season_distance(np.linspace(1990, 2010, 500), 2000.0)
    assert d.max() <= 0.5 + 1e-12
    assert d.min() >= 0.0

def test_season_distance_symmetric_about_target():
    before = season_distance(np.array([2000.0 - 0.2]), 2000.0)
    after = season_distance(np.array([2000.0 + 0.2]), 2000.0)
    assert before[0] == pytest.approx(after[0])

def test_season_distance_quarter_year_still_carries_weight():
    """A sample three months off-season must still be weighted.

    The superseded chord-distance metric gave zero weight beyond about
    29 days, which starved every grid point of data.
    """
    w = compute_weights(2000.0, 2.0, np.array([2000.25]), np.array([2.0]))
    assert w[0] > 0


# --- AdaptiveWeights ---------------------------------------------------
# EGRET's window expansion: widen until enough observations are in range.

@pytest.fixture
def sparse():
    years = np.linspace(1990.0, 2010.0, 40)
    logq = np.linspace(0.0, 4.0, 40)
    uncen = np.ones(40, dtype=int)
    return years, logq, uncen

def test_adaptive_weights_reaches_the_minimum(sparse):
    from pyegret.model.tri_cube import compute_adaptive_weights
    years, logq, uncen = sparse
    # Ask for 30 of the 40 observations at a far corner of the grid
    w, _ = compute_adaptive_weights(
        1990.0, 4.0, years, logq, uncen,
        min_num_obs=30, min_num_uncen=30,
    )
    assert np.sum((w > 0)) >= 30

def test_adaptive_weights_requirement_capped_at_record_length(sparse):
    """Asking for more observations than exist must still terminate."""
    from pyegret.model.tri_cube import compute_adaptive_weights
    years, logq, uncen = sparse
    w, _ = compute_adaptive_weights(
        2000.0, 2.0, years, logq, uncen,
        min_num_obs=10_000, min_num_uncen=10_000,
    )
    assert np.sum((w > 0)) == len(years)

def test_adaptive_weights_windows_widen_at_the_edges(sparse):
    """edge_adjust stretches the time window near the record ends."""
    from pyegret.model.tri_cube import compute_adaptive_weights
    years, logq, uncen = sparse
    _, (w_edge, _, _) = compute_adaptive_weights(
        1990.0, 2.0, years, logq, uncen, min_num_obs=1, min_num_uncen=1,
    )
    _, (w_mid, _, _) = compute_adaptive_weights(
        2000.0, 2.0, years, logq, uncen, min_num_obs=1, min_num_uncen=1,
    )
    assert w_edge > w_mid


# ─────────────────────────────────────────────────────────────────────────────
# run_surv_reg
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_sample():
    """50 uncensored synthetic observations."""
    n = 50
    dec_years = np.linspace(1990.0, 2010.0, n)
    logq = _RNG.normal(2.0, 0.8, n)
    log_conc = 0.3 + 0.1 * (dec_years - 2000) + 0.4 * logq + _RNG.normal(0, 0.2, n)
    return pd.DataFrame({
        "DecYear":     dec_years,
        "LogQ":        logq,
        "SinDY":       np.sin(_TWO_PI * dec_years),
        "CosDY":       np.cos(_TWO_PI * dec_years),
        "LogConcHigh": log_conc,
        "Uncen":       np.ones(n, dtype=int),
    })


@pytest.fixture(scope="module")
def mixed_sample(clean_sample):
    """Same as clean_sample but ~20% censored."""
    df = clean_sample.copy()
    mask = _RNG.uniform(size=len(df)) < 0.2
    df.loc[mask, "Uncen"] = 0
    return df


# --- RunSurvReg --------------------------------------------------------

def test_run_surv_reg_returns_finite_uncensored(clean_sample):
    weights = np.ones(len(clean_sample))
    yHat, SE, ConcHat = run_surv_reg(
        2000.0, 2.0, 0.0, 1.0, clean_sample, weights
    )
    assert np.isfinite(yHat)
    assert np.isfinite(SE)
    assert np.isfinite(ConcHat)

def test_run_surv_reg_conchat_positive(clean_sample):
    weights = np.ones(len(clean_sample))
    _, _, ConcHat = run_surv_reg(2000.0, 2.0, 0.0, 1.0, clean_sample, weights)
    assert ConcHat > 0

def test_run_surv_reg_se_positive(clean_sample):
    weights = np.ones(len(clean_sample))
    _, SE, _ = run_surv_reg(2000.0, 2.0, 0.0, 1.0, clean_sample, weights)
    assert SE > 0

def test_run_surv_reg_nan_on_zero_weights(clean_sample):
    weights = np.zeros(len(clean_sample))
    yHat, SE, ConcHat = run_surv_reg(2000.0, 2.0, 0.0, 1.0, clean_sample, weights)
    assert np.isnan(yHat)
    assert np.isnan(SE)
    assert np.isnan(ConcHat)

def test_run_surv_reg_nan_on_tiny_weights(clean_sample):
    weights = np.full(len(clean_sample), 1e-8)
    yHat, SE, ConcHat = run_surv_reg(2000.0, 2.0, 0.0, 1.0, clean_sample, weights)
    assert np.isnan(yHat)

def test_run_surv_reg_returns_finite_with_censored(mixed_sample):
    weights = np.ones(len(mixed_sample))
    yHat, SE, ConcHat = run_surv_reg(
        2000.0, 2.0, 0.0, 1.0, mixed_sample, weights
    )
    assert np.isfinite(yHat)
    assert np.isfinite(ConcHat)

def test_run_surv_reg_conchat_back_transform_consistent(clean_sample):
    """ConcHat must equal exp(yHat + 0.5 * SE²)."""
    weights = np.ones(len(clean_sample))
    yHat, SE, ConcHat = run_surv_reg(2000.0, 2.0, 0.0, 1.0, clean_sample, weights)
    expected = np.exp(yHat + 0.5 * SE**2)
    assert ConcHat == pytest.approx(expected, rel=1e-6)

def test_run_surv_reg_tricube_weights_accepted(clean_sample):
    """Non-uniform tricube weights should still give finite results."""
    weights = np.linspace(0.1, 1.0, len(clean_sample))
    yHat, SE, ConcHat = run_surv_reg(2000.0, 2.0, 0.0, 1.0, clean_sample, weights)
    assert np.isfinite(yHat)


# ─────────────────────────────────────────────────────────────────────────────
# est_surfaces  (tiny 4×3 grid to keep tests fast)
# ─────────────────────────────────────────────────────────────────────────────

# --- EstSurfaces -------------------------------------------------------

def test_est_surfaces_shape(synthetic_elist):
    import copy
    el = copy.deepcopy(synthetic_elist)
    surf = est_surfaces(el, n_vector_year=4, n_vector_logQ=3, verbose=False)
    assert surf.shape == (4, 3, 3)

def test_est_surfaces_stores_vector_year(synthetic_elist):
    import copy
    el = copy.deepcopy(synthetic_elist)
    est_surfaces(el, n_vector_year=4, n_vector_logQ=3, verbose=False)
    assert el.get("vector_year") is not None
    assert len(el["vector_year"]) == 4

def test_est_surfaces_stores_vector_logq(synthetic_elist):
    import copy
    el = copy.deepcopy(synthetic_elist)
    est_surfaces(el, n_vector_year=4, n_vector_logQ=3, verbose=False)
    assert el.get("vector_logQ") is not None
    assert len(el["vector_logQ"]) == 3

def test_est_surfaces_vector_year_spans_sample(synthetic_elist):
    import copy
    el = copy.deepcopy(synthetic_elist)
    est_surfaces(el, n_vector_year=4, n_vector_logQ=3, verbose=False)
    # EGRET runs the year axis over the whole years bracketing the record
    assert el["vector_year"][0] == pytest.approx(
        float(np.floor(el["Sample"]["DecYear"].min()))
    )
    assert el["vector_year"][-1] == pytest.approx(
        float(np.ceil(el["Sample"]["DecYear"].max()))
    )

def test_est_surfaces_slice_indices(synthetic_elist):
    """Most grid points should produce finite yHat."""
    import copy
    el = copy.deepcopy(synthetic_elist)
    surf = est_surfaces(el, n_vector_year=4, n_vector_logQ=3, verbose=False)
    # At least half the grid should be non-NaN
    assert np.sum(np.isfinite(surf[:, :, 0])) >= 6

def test_est_surfaces_conchat_positive_where_finite(synthetic_elist):
    import copy
    el = copy.deepcopy(synthetic_elist)
    surf = est_surfaces(el, n_vector_year=4, n_vector_logQ=3, verbose=False)
    conc_hat = surf[:, :, 2]
    finite_mask = np.isfinite(conc_hat)
    assert np.all((conc_hat[finite_mask] > 0))

def test_est_surfaces_raises_without_sample():
    info = pd.DataFrame({"station_nm": ["X"], "site_no": ["0"]})
    daily = pd.DataFrame({"Date": pd.to_datetime(["2000-01-01"]), "Q": [1.0]})
    el = make_elist(INFO=info, Daily=daily)
    with pytest.raises(ValueError, match="no Sample"):
        est_surfaces(el)


# ─────────────────────────────────────────────────────────────────────────────
# est_daily_from_surfaces
# ─────────────────────────────────────────────────────────────────────────────

# --- EstDaily ----------------------------------------------------------

def test_est_daily_adds_concday(modeled_elist):
    assert "ConcDay" in modeled_elist["Daily"].columns

def test_est_daily_adds_fluxday(modeled_elist):
    assert "FluxDay" in modeled_elist["Daily"].columns

def test_est_daily_adds_fnconc(modeled_elist):
    assert "FNConc" in modeled_elist["Daily"].columns

def test_est_daily_adds_fnflux(modeled_elist):
    assert "FNFlux" in modeled_elist["Daily"].columns

def test_est_daily_concday_positive(modeled_elist):
    vals = modeled_elist["Daily"]["ConcDay"].dropna()
    assert np.all((vals > 0))

def test_est_daily_fluxday_formula(modeled_elist):
    """FluxDay must equal ConcDay × Q × 86.4 everywhere (ignoring NaN)."""
    d = modeled_elist["Daily"]
    expected = d["ConcDay"] * d["Q"] * 86.4
    assert np.allclose(d["FluxDay"], expected, rtol=1e-6, equal_nan=True)

def test_est_daily_fnflux_is_positive(modeled_elist):
    assert np.all((modeled_elist["Daily"]["FNFlux"].dropna() > 0))

def test_est_daily_fnflux_is_not_fnconc_times_q(modeled_elist):
    """FNFlux averages conc x Q over the reference flows for that day of
    the year; it is not FNConc x Q for the day itself.

    The two coincide only if concentration and discharge are independent,
    which they never are.
    """
    d = modeled_elist["Daily"]
    naive = d["FNConc"] * d["Q"] * 86.4
    assert not np.allclose(d["FNFlux"], naive, rtol=1e-3, equal_nan=True)

def test_est_daily_fnconc_length(modeled_elist):
    assert len(modeled_elist["Daily"]["FNConc"]) == len(modeled_elist["Daily"])

def test_est_daily_raises_without_surfaces(synthetic_elist):
    import copy
    el = copy.deepcopy(synthetic_elist)
    with pytest.raises(ValueError, match="surfaces is None"):
        est_daily_from_surfaces(el)


# ─────────────────────────────────────────────────────────────────────────────
# model_estimation (integration)
# ─────────────────────────────────────────────────────────────────────────────

# --- ModelEstimation ---------------------------------------------------

def test_model_estimation_returns_elist(modeled_elist):
    assert isinstance(modeled_elist, dict)

def test_model_estimation_has_model(modeled_elist):
    assert has_model(modeled_elist)

def test_model_estimation_surfaces_shape(modeled_elist):
    # Year axis: 2000-01-01..2001-12-30 -> whole years 2000..2002,
    # at 16 points per year, plus the closing endpoint.
    assert modeled_elist["surfaces"].shape == (33, 14, 3)

def test_model_estimation_surface_has_no_nan_cells(modeled_elist):
    """Window expansion should leave every grid point estimable."""
    assert np.all(np.isfinite(modeled_elist["surfaces"][:, :, 0]))

def test_model_estimation_sample_has_yhat(modeled_elist):
    assert "yHat" in modeled_elist["Sample"].columns

def test_model_estimation_sample_has_se(modeled_elist):
    assert "SE" in modeled_elist["Sample"].columns

def test_model_estimation_sample_has_conchat(modeled_elist):
    assert "ConcHat" in modeled_elist["Sample"].columns

def test_model_estimation_sample_has_resid(modeled_elist):
    assert "resid" in modeled_elist["Sample"].columns

def test_model_estimation_raises_without_sample():
    info = pd.DataFrame({"station_nm": ["X"], "site_no": ["0"]})
    daily = pd.DataFrame({"Date": pd.to_datetime(["2000-01-01"]), "Q": [1.0]})
    el = make_elist(INFO=info, Daily=daily)
    with pytest.raises(ValueError, match="no Sample"):
        model_estimation(el)

def test_model_estimation_daily_row_count_unchanged(synthetic_elist, modeled_elist):
    assert len(modeled_elist["Daily"]) == len(synthetic_elist["Daily"])

def test_model_estimation_sample_row_count_unchanged(synthetic_elist, modeled_elist):
    assert len(modeled_elist["Sample"]) == len(synthetic_elist["Sample"])

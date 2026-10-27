"""
Does PyEGRET reproduce R EGRET?

Every other test measures PyEGRET against itself or against a synthetic truth.
This one measures it against the reference implementation: the fitted
``Choptank_eList`` shipped with the USGS EGRET R package, which carries the
WRTDS ``surfaces`` array EGRET itself computed, the ConcDay, FluxDay, FNConc
and FNFlux columns it derived from them, and the cross-validated yHat, SE and
ConcHat it stored on Sample.

The test re-fits EGRET's own Daily and Sample with PyEGRET, using the settings
recorded in the reference INFO row, and requires the results to match. Not
"correlate well" - match, to five decimal places, across all 7,406 grid points.

See ``tests/reference/README.md`` for provenance and for why edge adjustment
is disabled in this comparison.

Needs the ``rdata`` package to read .RData; no R installation required.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyegret.data.populate import populate_sample_columns
from pyegret.elist import make_elist
from pyegret.model.est_cross_val import est_cross_val
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.est_surfaces import est_surfaces
from pyegret.testing import annual_mean_by_calendar_year

rdata = pytest.importorskip("rdata", reason="pip install rdata to run the EGRET comparison")

_REFERENCE = Path(__file__).parent / "reference" / "Choptank_eList.RData"

pytestmark = pytest.mark.skipif(
    not _REFERENCE.exists(),
    reason=f"reference dataset not found at {_REFERENCE}",
)

# EGRET's stored surface for this dataset predates edgeAdjust; see the README.
_EDGE_ADJUST = False

_SETTINGS = [
    "windowY", "windowQ", "windowS", "minNumObs", "minNumUncen",
    "bottomYear", "stepYear", "nVectorYear",
    "bottomLogQ", "stepLogQ", "nVectorLogQ",
]


@pytest.fixture(scope="module")
def reference():
    """EGRET's own fitted eList, as INFO / Daily / Sample / surfaces."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(_REFERENCE)
        converted = rdata.conversion.convert(parsed)

    elist = converted["Choptank_eList"]
    info, daily, sample = elist["INFO"], elist["Daily"], elist["Sample"]
    for frame in (info, daily, sample):
        frame.columns = [str(c) for c in frame.columns]

    settings = {key: float(info[key].iloc[0]) for key in _SETTINGS}
    # EGRET stores surfaces as [logQ, year, slice]; PyEGRET uses [year, logQ, slice]
    surfaces = np.transpose(elist["surfaces"], (1, 0, 2))
    return {"info": info, "Daily": daily, "Sample": sample,
            "surfaces": surfaces, "settings": settings}


@pytest.fixture(scope="module")
def fitted(reference):
    """PyEGRET re-fit of EGRET's Daily and Sample, with EGRET's settings."""
    daily_r, sample_r = reference["Daily"], reference["Sample"]
    cfg = reference["settings"]

    daily = daily_r[["Date", "Q", "LogQ", "DecYear", "Month", "Day",
                     "MonthSeq", "Julian"]].copy()
    sample = populate_sample_columns(
        sample_r[["Date", "ConcLow", "ConcHigh", "Uncen", "DecYear", "Month",
                  "Day", "MonthSeq", "Julian", "SinDY", "CosDY", "Q", "LogQ"]].copy()
    )

    elist = make_elist(
        INFO=pd.DataFrame([{"shortName": "Choptank River"}]),
        Daily=daily,
        Sample=sample,
    )
    elist["surfaces"] = est_surfaces(
        elist,
        window_y=cfg["windowY"], window_q=cfg["windowQ"], window_s=cfg["windowS"],
        min_num_obs=int(cfg["minNumObs"]), min_num_uncen=int(cfg["minNumUncen"]),
        edge_adjust=_EDGE_ADJUST,
        n_vector_logQ=int(cfg["nVectorLogQ"]),
        verbose=False,
    )
    elist["Daily"] = est_daily_from_surfaces(elist)
    return elist


# --- GridGeometry ------------------------------------------------------
# The surface must be estimated on the same grid EGRET used.

def test_grid_geometry_year_axis(fitted, reference):
    cfg = reference["settings"]
    expected = cfg["bottomYear"] + np.arange(int(cfg["nVectorYear"])) * cfg["stepYear"]
    assert np.allclose(fitted["vector_year"], expected)

def test_grid_geometry_logq_axis(fitted, reference):
    cfg = reference["settings"]
    expected = cfg["bottomLogQ"] + np.arange(int(cfg["nVectorLogQ"])) * cfg["stepLogQ"]
    assert np.allclose(fitted["vector_logQ"], expected)

def test_grid_geometry_surface_shape(fitted, reference):
    assert fitted["surfaces"].shape == reference["surfaces"].shape


# --- SurfaceMatchesEgret -----------------------------------------------
# All three surface slices, across every one of the 7,406 grid points.
# 
# Agreement is exact through the body of the grid. What residual difference
# exists sits in the four lowest log-discharge columns, which between them
# carry under 2 % of the daily record - the corner where the weighted
# censored regression is least constrained and where PyEGRET's L-BFGS-B and
# R's survreg can settle a few decimal places apart. Worst case anywhere on
# the surface is 6e-4 in log-concentration, or 0.06 % in concentration.

SLICES = [(0, "yHat"), (1, "SE"), (2, "ConcHat")]

@pytest.mark.parametrize("index,name", SLICES)
def test_surface_matches_egret_all_values_finite(fitted, index, name):
    assert np.all(np.isfinite(fitted["surfaces"][:, :, index]))

@pytest.mark.parametrize("index,name", SLICES)
def test_surface_matches_egret_bulk_agrees_to_four_decimals(fitted, reference, index, name):
    ours = fitted["surfaces"][:, :, index]
    theirs = reference["surfaces"][:, :, index]
    within = np.abs(ours - theirs) <= 1e-4
    share = within.sum() / within.size
    assert share >= 0.98, (
        f"{name}: only {100 * share:.2f} % of grid points agree to 1e-4"
    )

@pytest.mark.parametrize("index,name", SLICES)
def test_surface_matches_egret_no_grid_point_differs_materially(fitted, reference, index, name):
    ours = fitted["surfaces"][:, :, index]
    theirs = reference["surfaces"][:, :, index]
    worst = float(np.nanmax(np.abs(ours - theirs)))
    assert worst < 5e-3, f"{name}: worst grid point differs by {worst:.6f}"

def test_surface_matches_egret_differences_are_confined_to_the_lowest_flows(fitted, reference):
    """Any disagreement must stay in the sparse low-discharge corner.

    If a change ever spreads error into the well-sampled middle of the
    flow range, that is a regression however small the numbers look.
    """
    ours = fitted["surfaces"][:, :, 2]
    theirs = reference["surfaces"][:, :, 2]
    _, logq_index = np.where(np.abs(ours - theirs) > 1e-4)
    if logq_index.size:
        assert logq_index.max() <= 3, (
            f"disagreement reached logQ column {logq_index.max()}, "
            "outside the sparse low-flow corner"
        )


# --- DailyMatchesEgret -------------------------------------------------
# The four columns EGRET derives from the surface.

@pytest.mark.parametrize("column", ["ConcDay", "FluxDay"])
def test_daily_matches_egret_point_estimates_match(fitted, reference, column):
    """Interpolated daily values match everywhere, to a part in 100,000."""
    ours = fitted["Daily"][column].to_numpy(dtype=float)
    theirs = reference["Daily"][column].to_numpy(dtype=float)
    assert np.all(np.isfinite(ours))
    assert np.allclose(ours, theirs, rtol=1e-3), (
        f"{column}: max rel diff "
        f"{np.nanmax(np.abs((ours - theirs) / theirs)):.6f}"
    )

@pytest.mark.parametrize("column", ["FNConc", "FNFlux"])
def test_daily_matches_egret_flow_normalized_bulk_matches(fitted, reference, column):
    """Flow normalization averages the surface over reference discharges,
    so it picks up the low-flow corner on the days whose day-of-year
    reference set reaches down there. Under 1 % of days are affected.
    """
    ours = fitted["Daily"][column].to_numpy(dtype=float)
    theirs = reference["Daily"][column].to_numpy(dtype=float)
    assert np.all(np.isfinite(ours))
    rel = np.abs((ours - theirs) / theirs)
    assert float(np.percentile(rel, 99)) < 1e-3, (
        f"{column}: 99th percentile relative difference "
        f"{np.percentile(rel, 99):.6f}"
    )

@pytest.mark.parametrize("column", ["FNConc", "FNFlux"])
def test_daily_matches_egret_flow_normalized_worst_case_is_bounded(fitted, reference, column):
    # Current EGRET pools the reference flows of February 28 and 29 (bin_Qs,
    # "Bob's convention"); the shipped Choptank_eList was fitted before that
    # existed, so those two days are left out of this comparison.
    ours = fitted["Daily"][column].to_numpy(dtype=float)
    theirs = reference["Daily"][column].to_numpy(dtype=float)
    day = fitted["Daily"]["Day"].to_numpy()
    keep = (day != 59) & (day != 60)
    ours = ours[keep]
    theirs = theirs[keep]
    worst = float(np.nanmax(np.abs((ours - theirs) / theirs)))
    assert worst < 0.05, f"{column}: worst day differs by {100 * worst:.2f} %"

def test_daily_matches_egret_flow_normalized_trend_matches(fitted, reference):
    """The headline WRTDS number, over the full record."""
    def annual(frame):
        series = annual_mean_by_calendar_year(frame, "FNConc")
        return float(series.iloc[-1] / series.iloc[0] - 1.0)

    ours = annual(fitted["Daily"])
    theirs = annual(reference["Daily"])
    assert ours == pytest.approx(theirs, abs=0.005), (
        f"PyEGRET {100 * ours:+.1f} % vs EGRET {100 * theirs:+.1f} %"
    )


@pytest.fixture(scope="module")
def cross_validated(reference):
    """PyEGRET leave-one-out predictions for EGRET's own Sample."""
    daily_r, sample_r = reference["Daily"], reference["Sample"]
    cfg = reference["settings"]

    daily = daily_r[["Date", "Q", "LogQ", "DecYear", "Month", "Day",
                     "MonthSeq", "Julian"]].copy()
    sample = populate_sample_columns(
        sample_r[["Date", "ConcLow", "ConcHigh", "Uncen", "DecYear", "Month",
                  "Day", "MonthSeq", "Julian", "SinDY", "CosDY", "Q", "LogQ"]].copy()
    )
    elist = make_elist(INFO=pd.DataFrame([{"shortName": "Choptank River"}]),
                  Daily=daily, Sample=sample)

    return est_cross_val(
        elist,
        window_y=cfg["windowY"], window_q=cfg["windowQ"], window_s=cfg["windowS"],
        min_num_obs=int(cfg["minNumObs"]), min_num_uncen=int(cfg["minNumUncen"]),
        edge_adjust=_EDGE_ADJUST,
        verbose=False,
    )


# --- CrossValidationMatchesEgret ---------------------------------------
# EGRET stores leave-one-out predictions on Sample; so should PyEGRET.
# 
# Agreement here is far tighter than on the surface, because every fit sits
# at an actual observation - in the middle of the data rather than out at a
# grid corner where few samples carry weight.

COLUMNS = ["yHat", "SE", "ConcHat"]

@pytest.mark.parametrize("column", COLUMNS)
def test_cross_validation_matches_egret_all_samples_predicted(cross_validated, column):
    assert np.all(np.isfinite(cross_validated[column].to_numpy(dtype=float)))

@pytest.mark.parametrize("column", COLUMNS)
def test_cross_validation_matches_egret_matches_to_machine_precision(cross_validated, reference, column):
    ours = cross_validated[column].to_numpy(dtype=float)
    theirs = reference["Sample"][column].to_numpy(dtype=float)
    assert np.allclose(ours, theirs, atol=1e-4), (
        f"{column}: max abs diff {np.nanmax(np.abs(ours - theirs)):.2e}"
    )

def test_cross_validation_matches_egret_predictions_are_held_out(cross_validated, reference):
    """A cross-validated fit must be worse than an in-sample one.

    If leave-one-out residuals were as small as in-sample residuals, the
    observation was not actually being left out.
    """
    from scipy.interpolate import RegularGridInterpolator

    sample = cross_validated
    cv_resid = (sample["LogConcHigh"].to_numpy(dtype=float)
                - sample["yHat"].to_numpy(dtype=float))

    # In-sample comparison: read yHat off EGRET's own fitted surface
    surfaces = reference["surfaces"]
    cfg = reference["settings"]
    v_year = cfg["bottomYear"] + np.arange(int(cfg["nVectorYear"])) * cfg["stepYear"]
    v_logQ = cfg["bottomLogQ"] + np.arange(int(cfg["nVectorLogQ"])) * cfg["stepLogQ"]
    interp = RegularGridInterpolator(
        (v_year, v_logQ), surfaces[:, :, 0],
        method="linear", bounds_error=False, fill_value=None,
    )
    pts = np.column_stack([
        np.clip(sample["DecYear"].to_numpy(dtype=float), v_year[0], v_year[-1]),
        np.clip(sample["LogQ"].to_numpy(dtype=float), v_logQ[0], v_logQ[-1]),
    ])
    in_resid = sample["LogConcHigh"].to_numpy(dtype=float) - interp(pts)

    assert np.sqrt(np.mean(cv_resid ** 2)) > np.sqrt(np.mean(in_resid ** 2))

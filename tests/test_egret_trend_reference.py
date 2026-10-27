"""
run_pairs and run_series against results R EGRET itself produced.

The reference files come from EGRET's own vignette (vignettes/Enhancements.Rmd,
EGRET 3.0.12), where they were saved with saveRDS:

- ``pairResults.rds``:  runPairs(Choptank_eList, 1985, 2010, windowSide = 7)
- ``pairResults2.rds``: the same with flowBreak, Q1EndDate, wall,
  sample1EndDate, QStartDate, QEndDate, paStart = 4, paLong = 5
- ``eListOut.rds``:     runSeries(Green River chloride, windowSide = 12,
  flowBreak = TRUE, Q1EndDate = "1963-03-31", wall = TRUE,
  sample1EndDate = "1963-03-01")

No R installation is needed; ``rdata`` reads the files.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rdata = pytest.importorskip("rdata", reason="pip install rdata to run the EGRET comparison")

from pyegret import merge_report  # noqa: E402
from pyegret.analysis import run_pairs, run_series, setup_years  # noqa: E402
from pyegret.data.populate import populate_date_columns, populate_sample_columns  # noqa: E402

_REFERENCE = Path(__file__).parent / "reference"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))
from choptank_wrtds import load_choptank  # noqa: E402

_X_COLUMNS = ["x10", "x11", "x20", "x22"]


def _read(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rdata.conversion.convert(rdata.parser.parse_file(_REFERENCE / name))


def _r_dates(values):
    return pd.to_datetime(values, unit="D", origin="1970-01-01")


@pytest.fixture(scope="module")
def choptank():
    info, daily, sample = load_choptank()
    return merge_report(info, daily, sample)


def test_run_pairs_matches_egret(choptank):
    ours = run_pairs(choptank, 1985, 2010, window_side=7, verbose=False)["pairResults"]
    theirs = _read("pairResults.rds")
    rel = ours[_X_COLUMNS].to_numpy() / theirs[_X_COLUMNS].to_numpy() - 1
    assert np.max(np.abs(rel)) < 0.002
    # The headline numbers as R prints them: +39 %, CQTC +42 %, QTC -3.1 %.
    total = 100 * ours.loc["Conc", "TotalChange"] / ours.loc["Conc", "x11"]
    assert total == pytest.approx(100 * theirs.loc["Conc", "TotalChange"] / theirs.loc["Conc", "x11"],
                                  abs=0.2)


def test_run_pairs_with_wall_break_and_season_matches_egret(choptank):
    ours = run_pairs(choptank, 1985, 2010, window_side=7, flow_break=True,
                     q1_end_date="1995-05-31", wall=True, sample1_end_date="1995-05-31",
                     q_start_date="1979-10-01", q_end_date="2010-09-30",
                     pa_start=4, pa_long=5, verbose=False)["pairResults"]
    theirs = _read("pairResults2.rds")
    rel = ours[_X_COLUMNS].to_numpy() / theirs[_X_COLUMNS].to_numpy() - 1
    assert np.max(np.abs(rel)) < 0.002
    assert ours.loc["Conc", "QTC"] == pytest.approx(theirs.loc["Conc", "QTC"], abs=1e-3)


@pytest.fixture(scope="module")
def green_river():
    out = _read("eListOut.rds")
    daily_r = out["Daily"]
    sample_r = out["Sample"]
    for frame in (daily_r, sample_r):
        frame.columns = [str(c) for c in frame.columns]

    daily = daily_r[["Date", "Q"]].copy()
    daily["Date"] = _r_dates(daily["Date"])
    daily["LogQ"] = np.log(daily["Q"].to_numpy(dtype=float))
    daily = populate_date_columns(daily)

    sample = sample_r[["Date", "ConcLow", "ConcHigh", "Uncen"]].copy()
    sample["Date"] = _r_dates(sample["Date"])
    sample["ConcLow"] = sample["ConcLow"].fillna(0.0)
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)

    info = pd.DataFrame([{"shortName": "Green River near Greendale, UT",
                          "paramShortName": "Chloride"}])
    elist = merge_report(info, daily, sample)

    r_daily = daily_r.copy()
    r_daily["Date"] = _r_dates(r_daily["Date"])
    return elist, r_daily


def test_run_series_with_wall_and_flow_break_matches_egret(green_river):
    elist, r_daily = green_river
    ours = run_series(elist, window_side=12, flow_break=True, q1_end_date="1963-03-31",
                      wall=True, sample1_end_date="1963-03-01", verbose=False)
    annual_ours = setup_years(ours["Daily"])
    annual_r = setup_years(r_daily)
    assert len(annual_ours) == len(annual_r)
    for column in ("FNConc", "FNFlux", "Conc"):
        ours_values = annual_ours[column].to_numpy(dtype=float)
        r_values = annual_r[column].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(ours_values), np.isnan(r_values))
        rel = np.abs(ours_values / r_values - 1)
        assert np.nanmedian(rel) < 1e-5, column
        assert np.nanmax(rel) < 0.005, column


# ---------------------------------------------------------------------------
# A model fitted by current EGRET with today's defaults (windowY = 7,
# edgeAdjust = TRUE): total phosphorus, Potomac River at Chain Bridge,
# from EGRET's WRTDSK vignette (ChainBridge.TP.RData).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain_bridge():
    from pyegret import model_estimation

    converted = _read("ChainBridge.TP.RData")["eList"]
    daily_r = converted["Daily"]
    sample_r = converted["Sample"]
    for frame in (daily_r, sample_r):
        frame.columns = [str(c) for c in frame.columns]

    daily = daily_r[["Date", "Q"]].copy()
    daily["Date"] = _r_dates(daily["Date"])
    daily["LogQ"] = np.log(daily["Q"].to_numpy(dtype=float))
    daily = populate_date_columns(daily)
    sample = sample_r[["Date", "ConcLow", "ConcHigh", "Uncen"]].copy()
    sample["Date"] = _r_dates(sample["Date"])
    sample["ConcLow"] = sample["ConcLow"].fillna(0.0)
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)

    elist = merge_report(pd.DataFrame([{"shortName": "Potomac River at Chain Bridge"}]),
                         daily, sample)
    fitted = model_estimation(elist, verbose=False)
    r_daily = daily_r.copy()
    r_daily["Date"] = _r_dates(r_daily["Date"])
    return fitted, r_daily, sample_r


def test_current_defaults_daily_estimates_match_egret(chain_bridge):
    fitted, r_daily, _ = chain_bridge
    for column in ("yHat", "SE", "ConcDay", "FNConc", "FNFlux"):
        rel = np.abs(fitted["Daily"][column].to_numpy() / r_daily[column].to_numpy() - 1)
        assert np.nanmedian(rel) < 1e-3, column
        assert np.nanquantile(rel, 0.99) < 5e-3, column


def test_current_defaults_annual_flow_normalized_match_egret(chain_bridge):
    fitted, r_daily, _ = chain_bridge
    ours = setup_years(fitted["Daily"])
    theirs = setup_years(r_daily)
    for column in ("FNConc", "FNFlux", "Conc", "Flux"):
        rel = np.abs(ours[column].to_numpy() / theirs[column].to_numpy() - 1)
        assert np.nanmax(rel) < 1e-3, column


def test_current_defaults_cross_validation_matches_egret(chain_bridge):
    fitted, _, sample_r = chain_bridge
    diff = np.abs(fitted["Sample"]["yHat"].to_numpy() - sample_r["yHat"].to_numpy())
    assert np.nanmax(diff) < 1e-3

"""
Tests for the data quality layer.

The subject is censored data. A result reported as "< 0.05" is an interval,
not a number, and almost every way of flattening it into a number is wrong in
a way that biases the analysis. These tests pin down the representation and
the ways it can be damaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyegret.data.populate import populate_date_columns, populate_sample_columns
from pyegret.data.quality import (
    data_overview,
    fix_sample_frame,
    make_augmented_sample,
    populate_concentrations,
    process_qw_data,
)
from pyegret.elist import make_elist


# --- PopulateConcentrations --------------------------------------------

def test_populate_concentrations_detected_value_has_equal_bounds():
    out = populate_concentrations([1.5], [""])
    assert out["ConcLow"].iloc[0] == 1.5
    assert out["ConcHigh"].iloc[0] == 1.5
    assert out["Uncen"].iloc[0] == 1

def test_populate_concentrations_less_than_zeroes_the_lower_bound():
    """A non-detect means the value lies somewhere in [0, limit]."""
    out = populate_concentrations([0.05], ["<"])
    assert out["ConcLow"].iloc[0] == 0.0
    assert out["ConcHigh"].iloc[0] == 0.05
    assert out["Uncen"].iloc[0] == 0

def test_populate_concentrations_greater_than_zeroes_the_upper_bound():
    out = populate_concentrations([100.0], [">"])
    assert out["ConcLow"].iloc[0] == 100.0
    assert out["ConcHigh"].iloc[0] == 0.0
    assert out["Uncen"].iloc[0] == 0

def test_populate_concentrations_no_remark_column_means_all_detected():
    out = populate_concentrations([1.0, 2.0, 3.0])
    assert np.all((out["Uncen"] == 1))

def test_populate_concentrations_missing_values_become_zero():
    out = populate_concentrations([np.nan], [""])
    assert out["ConcHigh"].iloc[0] == 0.0

def test_populate_concentrations_whitespace_around_a_remark_is_ignored():
    out = populate_concentrations([0.05], ["  <  "])
    assert out["Uncen"].iloc[0] == 0

def test_populate_concentrations_mixed_record():
    out = populate_concentrations([1.0, 0.05, 2.0], ["", "<", None])
    assert out["Uncen"].tolist() == [1, 0, 1]


# --- ProcessQwData -----------------------------------------------------

@pytest.fixture
def lab_export():
    return pd.DataFrame({
        "Date": ["2001-03-04", "2001-01-02", "2001-02-03"],
        "value": [1.2, 0.05, 0.8],
        "remark_cd": ["", "<", ""],
    })

def test_process_qw_data_builds_the_egret_columns(lab_export):
    out = process_qw_data(lab_export)
    assert set(out.columns) == {"Date", "ConcLow", "ConcHigh", "Uncen"}

def test_process_qw_data_sorts_by_date(lab_export):
    out = process_qw_data(lab_export)
    assert out["Date"].is_monotonic_increasing

def test_process_qw_data_carries_the_censoring_through(lab_export):
    out = process_qw_data(lab_export)
    censored = out[out["Uncen"] == 0]
    assert len(censored) == 1
    assert censored["ConcHigh"].iloc[0] == pytest.approx(0.05)

def test_process_qw_data_undated_rows_are_dropped():
    frame = pd.DataFrame({
        "Date": ["2001-01-01", "not a date"],
        "value": [1.0, 2.0],
        "remark_cd": ["", ""],
    })
    assert len(process_qw_data(frame)) == 1

def test_process_qw_data_no_remark_column_is_allowed(lab_export):
    out = process_qw_data(lab_export.drop(columns=["remark_cd"]), remark_col=None)
    assert np.all((out["Uncen"] == 1))

def test_process_qw_data_missing_column_names_what_is_available(lab_export):
    with pytest.raises(KeyError, match="Available"):
        process_qw_data(lab_export, value_col="concentration")


def _sample_elist(conc_low, conc_high, uncen):
    sample = pd.DataFrame({
        "Date": pd.date_range("2000-01-01", periods=len(conc_high), freq="30D"),
        "ConcLow": conc_low,
        "ConcHigh": conc_high,
        "Uncen": uncen,
    })
    sample = populate_sample_columns(populate_date_columns(sample))
    daily = pd.DataFrame({
        "Date": pd.date_range("2000-01-01", periods=400, freq="D"),
        "Q": np.linspace(1.0, 5.0, 400),
    })
    daily["LogQ"] = np.log(daily["Q"])
    daily = populate_date_columns(daily)
    return make_elist(INFO=pd.DataFrame([{"shortName": "Test"}]), Daily=daily, Sample=sample)


# --- FixSampleFrame ----------------------------------------------------

def test_fix_sample_frame_recomputes_conc_ave_for_detections():
    elist = _sample_elist([1.0, 2.0], [1.0, 2.0], [1, 1])
    elist["Sample"]["ConcAve"] = 999.0
    fixed = fix_sample_frame(elist)
    assert fixed["Sample"]["ConcAve"].tolist() == [1.0, 2.0]

def test_fix_sample_frame_censored_row_averages_to_half_the_limit():
    elist = _sample_elist([0.0], [0.05], [0])
    fixed = fix_sample_frame(elist)
    assert fixed["Sample"]["ConcAve"].iloc[0] == pytest.approx(0.025)

def test_fix_sample_frame_accepts_nan_as_a_missing_lower_bound():
    """R EGRET spells a censored lower bound NA; PyEGRET spells it 0."""
    elist = _sample_elist([np.nan], [0.05], [0])
    fixed = fix_sample_frame(elist)
    assert fixed["Sample"]["ConcAve"].iloc[0] == pytest.approx(0.025)

def test_fix_sample_frame_repairs_a_wrong_uncen_flag():
    elist = _sample_elist([0.0], [0.05], [1])       # flag says detected
    fixed = fix_sample_frame(elist)
    assert fixed["Sample"]["Uncen"].iloc[0] == 0        # bounds say otherwise

def test_fix_sample_frame_raises_without_sample():
    elist = make_elist(INFO=pd.DataFrame([{}]), Daily=pd.DataFrame({"Q": [1.0]}))
    with pytest.raises(ValueError, match="no Sample"):
        fix_sample_frame(elist)


# --- DataOverview ------------------------------------------------------

def test_data_overview_reports_the_record_extents(capsys):
    elist = _sample_elist([1.0, 2.0], [1.0, 2.0], [1, 1])
    summary = data_overview(elist, verbose=False)
    assert summary["n_samples"] == 2
    assert summary["n_days"] == 400
    assert summary["problems"] == []

def test_data_overview_counts_censored_samples():
    elist = _sample_elist([0.0, 1.0, 2.0], [0.05, 1.0, 2.0], [0, 1, 1])
    summary = data_overview(elist, verbose=False)
    assert summary["n_censored"] == 1
    assert summary["pct_censored"] == pytest.approx(33.3, abs=0.1)

def test_data_overview_flags_samples_beyond_the_discharge_record():
    elist = _sample_elist([1.0], [1.0], [1])
    elist["Sample"].loc[0, "Date"] = pd.Timestamp("2020-01-01")
    summary = data_overview(elist, verbose=False)
    assert any("postdates" in problem for problem in summary["problems"])

def test_data_overview_flags_a_heavily_censored_record():
    elist = _sample_elist([0.0] * 9 + [1.0], [0.05] * 9 + [1.0], [0] * 9 + [1])
    summary = data_overview(elist, verbose=False)
    assert any("censored" in problem for problem in summary["problems"])

def test_data_overview_prints_when_asked(capsys):
    elist = _sample_elist([1.0], [1.0], [1])
    data_overview(elist, verbose=True)
    assert "Discharge record" in capsys.readouterr().out

def test_data_overview_accepts_frames_instead_of_an_elist():
    elist = _sample_elist([1.0], [1.0], [1])
    summary = data_overview(daily=elist["Daily"], sample=elist["Sample"], verbose=False)
    assert summary["n_samples"] == 1

def test_data_overview_needs_something_to_summarise():
    with pytest.raises(ValueError, match="Pass an eList"):
        data_overview()


# --- MakeAugmentedSample -----------------------------------------------

@pytest.fixture
def modelled():
    elist = _sample_elist(
        [0.0, 1.0, 0.0, 2.0], [0.05, 1.0, 0.10, 2.0], [0, 1, 0, 1]
    )
    elist["Sample"]["yHat"] = [-2.5, 0.0, -2.0, 0.7]
    elist["Sample"]["SE"] = [0.3, 0.3, 0.3, 0.3]
    return elist

def test_make_augmented_sample_adds_both_columns(modelled):
    out = make_augmented_sample(modelled, seed=1)
    assert "rResid" in out["Sample"].columns
    assert "rObserved" in out["Sample"].columns

def test_make_augmented_sample_detected_values_are_reproduced_exactly(modelled):
    """Nothing is drawn for a detection - the residual is known."""
    out = make_augmented_sample(modelled, seed=1)
    sample = out["Sample"]
    detected = sample["Uncen"] == 1
    assert np.allclose(
        sample.loc[detected, "rObserved"], sample.loc[detected, "ConcHigh"]
    )

def test_make_augmented_sample_drawn_values_respect_the_reporting_limit(modelled):
    """A drawn value for a non-detect must land inside [0, limit]."""
    out = make_augmented_sample(modelled, seed=1)
    sample = out["Sample"]
    censored = sample["Uncen"] == 0
    assert np.all((sample.loc[censored, "rObserved"] > 0))
    assert (sample.loc[censored, "rObserved"]
            <= sample.loc[censored, "ConcHigh"] + 1e-9).all()

def test_make_augmented_sample_seed_makes_the_draw_reproducible(modelled):
    import copy
    first = make_augmented_sample(copy.deepcopy(modelled), seed=42)["Sample"]["rObserved"]
    second = make_augmented_sample(copy.deepcopy(modelled), seed=42)["Sample"]["rObserved"]
    assert np.allclose(first, second)

def test_make_augmented_sample_different_seeds_give_different_draws(modelled):
    import copy
    first = make_augmented_sample(copy.deepcopy(modelled), seed=1)["Sample"]
    second = make_augmented_sample(copy.deepcopy(modelled), seed=2)["Sample"]
    censored = first["Uncen"] == 0
    assert not np.allclose(
        first.loc[censored, "rObserved"], second.loc[censored, "rObserved"]
    )

def test_make_augmented_sample_requires_the_model_to_have_been_run():
    elist = _sample_elist([1.0], [1.0], [1])
    with pytest.raises(ValueError, match="model_estimation"):
        make_augmented_sample(elist)

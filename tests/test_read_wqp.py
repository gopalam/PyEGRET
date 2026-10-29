"""
Tests for the Water Quality Portal reader and the processing it shares with
the USGS Samples reader.

The network call is replaced by a stand-in ``dataretrieval.wqp`` that returns
results shaped like the portal's legacy ``resultPhysChem`` profile (column
names with slashes, as Python's dataretrieval gives them). The rows mimic real
Washington records checked live in October 2026: a Nooksack Tribe site that
reports "Not Detected" with a detection limit, and a Thurston County site that
reports "Below Detection Limit" with none.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from pyegret.data import read_wqp
from pyegret.data.qw_processing import process_wqp_results, wqp_results_to_sample


def _portal_results():
    return pd.DataFrame({
        "MonitoringLocationIdentifier": ["NOOKSACK_WQX-SW08"] * 5,
        "ActivityStartDate": ["2014-05-13", "2014-06-10", "2014-07-09", "2014-08-12",
                              "2014-09-09"],
        "ActivityTypeCode": ["Sample-Routine"] * 4 + ["Quality Control Sample-Field Replicate"],
        "ActivityMediaName": ["Water"] * 5,
        "ActivityMediaSubdivisionName": ["Surface Water"] * 5,
        "CharacteristicName": ["Phosphorus"] * 5,
        "ResultSampleFractionText": ["Dissolved"] * 5,
        "ResultDetectionConditionText": ["Not Detected", np.nan, "Below Detection Limit",
                                         np.nan, "Below Detection Limit"],
        "ResultMeasureValue": [np.nan, 0.031, np.nan, 0.027, np.nan],
        "ResultMeasure/MeasureUnitCode": [np.nan, "mg/L", np.nan, "mg/L", np.nan],
        "DetectionQuantitationLimitMeasure/MeasureValue": [0.01, np.nan, 0.005, np.nan, np.nan],
        "ResultAnalyticalMethod/MethodIdentifier": ["365.1"] * 5,
        "SampleCollectionMethod/MethodName": ["Grab"] * 5,
    })


@pytest.fixture
def stub_wqp(monkeypatch):
    """Install a stand-in dataretrieval.wqp that records its query."""
    calls = {"response": _portal_results()}

    def get_results(**kwargs):
        calls["query"] = kwargs
        return calls["response"], None

    wqp = types.SimpleNamespace(get_results=get_results)
    package = types.ModuleType("dataretrieval")
    package.wqp = wqp
    monkeypatch.setitem(sys.modules, "dataretrieval", package)
    monkeypatch.setitem(sys.modules, "dataretrieval.wqp", wqp)
    return calls


# --- processQWData -----------------------------------------------------

def test_non_detect_takes_its_detection_limit():
    """The bug fixed in 0.1.1: non-detects used to be dropped."""
    processed = process_wqp_results(_portal_results(), verbose=False)
    assert processed["qualifier"].iloc[0] == "<"
    assert processed["value"].iloc[0] == pytest.approx(0.01)


def test_below_detection_limit_is_censored_too():
    """Not in R's list; PyEGRET censors it rather than dropping it."""
    processed = process_wqp_results(_portal_results(), verbose=False)
    assert processed["qualifier"].iloc[2] == "<"
    assert processed["value"].iloc[2] == pytest.approx(0.005)


def test_measured_values_are_kept_as_reported():
    processed = process_wqp_results(_portal_results(), verbose=False)
    assert list(processed["qualifier"].iloc[[1, 3]]) == ["", ""]
    assert list(processed["value"].iloc[[1, 3]]) == [0.031, 0.027]


def test_slash_and_dot_column_names_are_the_same():
    slashes = process_wqp_results(_portal_results(), verbose=False)
    dotted = _portal_results()
    dotted.columns = [name.replace("/", ".") for name in dotted.columns]
    dots = process_wqp_results(dotted, verbose=False)
    assert np.allclose(slashes["value"], dots["value"], equal_nan=True)


def test_a_non_detect_without_a_limit_is_named_and_dropped():
    with pytest.warns(UserWarning, match="no detection limit"):
        sample = wqp_results_to_sample(_portal_results(), verbose=False)
    assert len(sample) == 4
    assert pd.Timestamp("2014-09-09") not in set(sample["Date"])


# --- read_wqp_sample ---------------------------------------------------

def test_read_wqp_sample_builds_a_censored_sample(stub_wqp):
    with pytest.warns(UserWarning):
        sample = read_wqp.read_wqp_sample("NOOKSACK_WQX-SW08", "Phosphorus", verbose=False)
    censored = sample[sample["Uncen"] == 0]
    assert len(censored) == 2
    assert list(censored["ConcHigh"]) == [0.01, 0.005]
    assert np.all(censored["ConcLow"] == 0.0)
    assert sample["Date"].is_monotonic_increasing


def test_read_wqp_sample_query_matches_r(stub_wqp):
    with pytest.warns(UserWarning):
        read_wqp.read_wqp_sample("NOOKSACK_WQX-SW08", "Phosphorus", "2010-01-01",
                                 "2020-12-31", verbose=False)
    query = stub_wqp["query"]
    assert query["siteid"] == "NOOKSACK_WQX-SW08"
    assert query["characteristicName"] == "Phosphorus"
    assert query["dataProfile"] == "resultPhysChem"
    assert query["legacy"] is True
    # The portal wants MM-DD-YYYY.
    assert query["startDateLo"] == "01-01-2010"
    assert query["startDateHi"] == "12-31-2020"


def test_read_wqp_sample_rejects_a_date_it_cannot_read(stub_wqp):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        read_wqp.read_wqp_sample("NOOKSACK_WQX-SW08", "Phosphorus", "01/01/2010")


def test_read_wqp_sample_keeps_the_descriptive_columns(stub_wqp, capsys):
    with pytest.warns(UserWarning):
        sample = read_wqp.read_wqp_sample("NOOKSACK_WQX-SW08", "Phosphorus")
    assert "ActivityTypeCode" in sample.columns
    assert "ResultSampleFraction" in sample.columns
    printed = capsys.readouterr()
    assert "5 samples retrieved." in printed.out
    assert "ActivityTypeCode" in printed.out


def test_read_wqp_sample_empty_response_raises(stub_wqp):
    stub_wqp["response"] = pd.DataFrame()
    with pytest.raises(ValueError, match="case-sensitive"):
        read_wqp.read_wqp_sample("NOOKSACK_WQX-SW08", "phosphorus")


def test_read_wqp_sample_sends_usgs_sites_to_the_samples_service(monkeypatch):
    """As R readWQPSample does."""
    from pyegret.data import read_waterdata

    seen = {}

    def fake(site_id, parameter_cd, start_date="", end_date="", **kwargs):
        seen["args"] = (site_id, parameter_cd, start_date, end_date)
        return "from the Samples service"

    monkeypatch.setattr(read_waterdata, "read_waterdata_sample", fake)
    result = read_wqp.read_wqp_sample("USGS-01491000", "00631", "2000-01-01", verbose=False)
    assert result == "from the Samples service"
    assert seen["args"] == ("USGS-01491000", "00631", "2000-01-01", "")

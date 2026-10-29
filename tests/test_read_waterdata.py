"""
Tests for the modern USGS Water Data API retrieval.

The network call itself is stubbed, and everything around it - query
construction, unit handling, censoring, column mapping - is tested against
synthetic responses shaped like the real ones (the Samples response mirrors
what the live service returned in October 2026).

``diagnostics/check_waterdata.py`` checks the readers against the live API and
compares the result with EGRET's own copy of the Choptank record; rerun it
when USGS or dataretrieval change.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from pyegret.data import read_waterdata as rw
from pyegret.data.qw_processing import process_wqp_results


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

# --- LocationId --------------------------------------------------------

def test_location_id_adds_agency_prefix():
    assert rw.usgs_location_id("01491000") == "USGS-01491000"

def test_location_id_leaves_prefixed_id_alone():
    assert rw.usgs_location_id("USGS-01491000") == "USGS-01491000"

def test_location_id_strips_whitespace():
    assert rw.usgs_location_id("  01491000 ") == "USGS-01491000"

def test_location_id_accepts_other_agencies():
    assert rw.usgs_location_id("EPA-12345") == "EPA-12345"


# --- TimeInterval ------------------------------------------------------

def test_time_interval_bounded():
    assert rw._time_interval("1979-10-01", "2011-09-30") == "1979-10-01/2011-09-30"

def test_time_interval_open_start():
    assert rw._time_interval("", "2011-09-30") == "../2011-09-30"

def test_time_interval_open_end():
    assert rw._time_interval("1979-10-01", "") == "1979-10-01/.."

def test_time_interval_unbounded_is_none():
    assert rw._time_interval("", "") is None


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# --- UnitDetection -----------------------------------------------------

def test_unit_detection_reads_cubic_feet_from_the_service():
    frame = pd.DataFrame({"value": [1.0], "unit_of_measure": ["ft3/s"]})
    assert rw._is_cubic_feet(frame, "00060") is True

def test_unit_detection_reads_cubic_metres_from_the_service():
    frame = pd.DataFrame({"value": [1.0], "unit_of_measure": ["m3/s"]})
    assert rw._is_cubic_feet(frame, "00060") is False

def test_unit_detection_falls_back_to_parameter_code():
    """00060 is defined in ft3/s, so assume that when no unit is given."""
    frame = pd.DataFrame({"value": [1.0]})
    assert rw._is_cubic_feet(frame, "00060") is True
    assert rw._is_cubic_feet(frame, "00065") is False

def test_unit_detection_warns_on_an_unrecognised_unit():
    frame = pd.DataFrame({"value": [1.0], "unit_of_measure": ["furlongs/fortnight"]})
    with pytest.warns(RuntimeWarning, match="Unrecognised discharge unit"):
        rw._is_cubic_feet(frame, "00060")


# ---------------------------------------------------------------------------
# Daily values
# ---------------------------------------------------------------------------

def _stub_daily_response(units="ft3/s", n=5):
    return pd.DataFrame({
        "monitoring_location_id": ["USGS-01491000"] * n,
        "parameter_code": ["00060"] * n,
        "statistic_id": ["00003"] * n,
        "time": [str(day.date()) for day in pd.date_range("2000-01-01", periods=n, freq="D")],
        "value": np.linspace(100.0, 140.0, n),
        "unit_of_measure": [units] * n,
        "qualifier": ["A"] * n,
    })


@pytest.fixture
def stub_waterdata(monkeypatch):
    """Replace the network layer with a recorder that returns canned frames."""
    calls = {}

    def get_daily(**kwargs):
        calls["daily"] = kwargs
        return calls.get("_daily_response", _stub_daily_response())

    def get_samples(**kwargs):
        calls["samples"] = kwargs
        return calls.get("_samples_response", pd.DataFrame())

    def get_monitoring_locations(**kwargs):
        calls["info"] = kwargs
        return calls.get("_info_response", pd.DataFrame())

    # A stand-in for the dataretrieval.waterdata module: a namespace holding
    # the functions PyEGRET calls.
    def get_reference_table(collection, query=None, **kwargs):
        calls["reference"] = (collection, query)
        return calls.get("_reference_response", pd.DataFrame())

    stub = types.SimpleNamespace(get_daily=get_daily, get_samples=get_samples,
                                 get_monitoring_locations=get_monitoring_locations,
                                 get_reference_table=get_reference_table)

    def import_stub():
        return stub

    monkeypatch.setattr(rw, "_import_waterdata", import_stub)
    return calls


# --- ReadDaily ---------------------------------------------------------

def test_read_daily_query_uses_modern_parameter_names(stub_waterdata):
    rw.read_waterdata_daily("01491000", "00060", "1979-10-01", "2011-09-30")
    query = stub_waterdata["daily"]
    assert query["monitoring_location_id"] == "USGS-01491000"
    assert query["parameter_code"] == "00060"
    assert query["statistic_id"] == "00003"
    assert query["time"] == "1979-10-01/2011-09-30"

def test_read_daily_omits_time_when_unbounded(stub_waterdata):
    rw.read_waterdata_daily("01491000")
    assert "time" not in stub_waterdata["daily"]

def test_read_daily_converts_cubic_feet_to_cubic_metres(stub_waterdata):
    daily = rw.read_waterdata_daily("01491000", "00060")
    assert daily["Q"].iloc[0] == pytest.approx(100.0 * rw.CFS_TO_CMS)

def test_read_daily_leaves_cubic_metres_alone(stub_waterdata):
    stub_waterdata["_daily_response"] = _stub_daily_response(units="m3/s")
    daily = rw.read_waterdata_daily("01491000", "00060")
    assert daily["Q"].iloc[0] == pytest.approx(100.0)

def test_read_daily_convert_false_skips_conversion(stub_waterdata):
    daily = rw.read_waterdata_daily("01491000", "00060", convert=False)
    assert daily["Q"].iloc[0] == pytest.approx(100.0)

def test_read_daily_produces_egret_daily_columns(stub_waterdata):
    daily = rw.read_waterdata_daily("01491000")
    for column in ("Date", "Q", "LogQ", "DecYear", "Month", "Day",
                   "MonthSeq", "Julian", "SinDY", "CosDY"):
        assert column in daily.columns

def test_read_daily_handles_zero_and_missing_like_egret(stub_waterdata):
    # populateDaily keeps every calendar day: a missing value stays NaN, and
    # a zero flow triggers the 0.001 * mean(Q) shift so the log is defined.
    response = _stub_daily_response()
    response.loc[0, "value"] = 0.0
    response.loc[1, "value"] = np.nan
    stub_waterdata["_daily_response"] = response
    daily = rw.read_waterdata_daily("01491000", verbose=False)
    assert len(daily) == 5
    assert np.isnan(daily["Q"].iloc[1])
    assert daily["Q"].iloc[0] > 0

def test_read_daily_deduplicates_dates(stub_waterdata):
    response = pd.concat([_stub_daily_response(), _stub_daily_response()])
    stub_waterdata["_daily_response"] = response
    daily = rw.read_waterdata_daily("01491000")
    assert daily["Date"].is_unique

def test_read_daily_accepts_a_tuple_response(stub_waterdata):
    stub_waterdata["_daily_response"] = (_stub_daily_response(), {"meta": 1})
    daily = rw.read_waterdata_daily("01491000")
    assert len(daily) == 5

def test_read_daily_raw_returns_the_untouched_frame(stub_waterdata):
    raw = rw.read_waterdata_daily("01491000", raw=True)
    assert "unit_of_measure" in raw.columns

def test_read_daily_empty_response_raises(stub_waterdata):
    stub_waterdata["_daily_response"] = pd.DataFrame()
    with pytest.raises(ValueError, match="No daily values"):
        rw.read_waterdata_daily("01491000")

def test_read_daily_unknown_columns_name_what_came_back(stub_waterdata):
    stub_waterdata["_daily_response"] = pd.DataFrame({"surprise": [1.0]})
    with pytest.raises(ValueError, match="available columns are"):
        rw.read_waterdata_daily("01491000")


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def _stub_samples_response():
    """Shaped like the live Samples service (fullphyschem), October 2026."""
    return pd.DataFrame({
        "Location_Identifier": ["USGS-01491000"] * 5,
        "Activity_StartDate": ["2000-01-05", "2000-02-09", "2000-03-15", "2000-04-19",
                               "2000-01-05"],
        "Activity_StartDateTime": ["2000-01-05 15:30:00+00:00", "2000-02-09 16:00:00+00:00",
                                   "2000-03-15 15:00:00+00:00", "2000-04-19 15:45:00+00:00",
                                   "2000-01-05 15:45:00+00:00"],
        "Activity_TypeCode": ["Sample-Routine"] * 4 + ["Sample-Integrated Time Series"],
        "Activity_Media": ["Water"] * 5,
        "Activity_MediaSubdivision": ["Surface water"] * 5,
        "Result_Characteristic": ["Inorganic nitrogen (nitrate and nitrite)"] * 5,
        "Result_Measure": [1.20, 0.85, np.nan, 1.05, 1.25],
        "Result_MeasureUnit": ["mg/L", "mg/L", np.nan, "mg/L", "mg/L"],
        "Result_ResultDetectionCondition": [np.nan, np.nan, "Not Detected", np.nan, np.nan],
        "DetectionLimit_MeasureA": [np.nan, np.nan, 0.05, np.nan, np.nan],
        "USGSpcode": ["00631"] * 5,
    })


# --- ReadSample --------------------------------------------------------

@pytest.fixture
def canned(stub_waterdata):
    """The stubbed service, primed with a four-row Samples response."""
    stub_waterdata["_samples_response"] = _stub_samples_response()
    return stub_waterdata

def test_read_sample_query_uses_samples_service_names(canned):
    rw.read_waterdata_sample("01491000", "00631", "1979-10-01", "2011-09-30")
    query = canned["samples"]
    assert query["monitoring_location_id"] == "USGS-01491000"
    assert query["usgs_pcode"] == "00631"
    assert query["service"] == "results"
    assert query["profile"] == "fullphyschem"
    assert query["activity_start_date_lower"] == "1979-10-01"
    assert query["activity_start_date_upper"] == "2011-09-30"


def test_read_sample_a_name_is_queried_as_a_characteristic(canned):
    """As in R: anything but a five-digit code is a characteristic name."""
    rw.read_waterdata_sample("01491000", "Phosphorus")
    query = canned["samples"]
    assert query["characteristic"] == "Phosphorus"
    assert "usgs_pcode" not in query


def test_read_sample_keeps_every_result_as_r_does(canned):
    """R keeps several results on one day; it does not average or drop them."""
    sample = rw.read_waterdata_sample("01491000", "00631", verbose=False)
    assert len(sample) == 5
    same_day = sample[sample["Date"] == pd.Timestamp("2000-01-05")]
    assert list(same_day["ConcHigh"]) == [1.20, 1.25]


def test_read_sample_carries_the_descriptive_columns(canned):
    sample = rw.read_waterdata_sample("01491000", "00631", verbose=False)
    for column in ("ActivityTypeCode", "ActivityMediaName", "ActivityStartDateTime",
                   "CharacteristicName", "USGSPCode"):
        assert column in sample.columns
    assert set(sample["ActivityTypeCode"]) == {"Sample-Routine",
                                               "Sample-Integrated Time Series"}


def test_read_sample_reports_characteristics_that_vary(canned, capsys):
    rw.read_waterdata_sample("01491000", "00631")
    printed = capsys.readouterr()
    assert "5 samples retrieved." in printed.out
    assert "Multiple values for some sample characteristics" in printed.out
    assert "ActivityTypeCode" in printed.out


def test_read_sample_recognises_r_non_detect_wordings():
    """processQWData marks these, in any case, as "<"."""
    texts = ["Not Detected", "Non-Detect", "non detect", "Detected Not Quantified",
             "Below Quantification Limit", np.nan]
    frame = pd.DataFrame({
        "ActivityStartDate": ["2000-01-0" + str(i + 1) for i in range(6)],
        "ResultMeasureValue": [np.nan] * 5 + [0.7],
        "ResultDetectionConditionText": texts,
        "DetectionQuantitationLimitMeasure.MeasureValue": [0.05] * 5 + [np.nan],
    })
    processed = process_wqp_results(frame, verbose=False)
    assert list(processed["qualifier"]) == ["<"] * 5 + [""]
    assert list(processed["value"]) == [0.05] * 5 + [0.7]

def test_read_sample_detected_values_are_uncensored(canned):
    sample = rw.read_waterdata_sample("01491000", "00631")
    detected = sample[sample["Uncen"] == 1]
    assert len(detected) == 4
    assert np.all((detected["ConcLow"] == detected["ConcHigh"]))

def test_read_sample_non_detect_uses_the_reporting_limit(canned):
    sample = rw.read_waterdata_sample("01491000", "00631")
    censored = sample[sample["Uncen"] == 0]
    assert len(censored) == 1
    assert censored["ConcHigh"].iloc[0] == pytest.approx(0.05)
    assert censored["ConcLow"].iloc[0] == 0.0

def test_read_sample_non_detect_conc_ave_is_half_the_limit(canned):
    """EGRET's convention: a censored 0.05 result averages to 0.025."""
    sample = rw.read_waterdata_sample("01491000", "00631")
    censored = sample[sample["Uncen"] == 0]
    assert censored["ConcAve"].iloc[0] == pytest.approx(0.025)

def test_read_sample_produces_egret_sample_columns(canned):
    sample = rw.read_waterdata_sample("01491000", "00631")
    for column in ("Date", "ConcLow", "ConcHigh", "Uncen", "ConcAve",
                   "LogConcLow", "LogConcHigh", "DecYear", "SinDY", "CosDY"):
        assert column in sample.columns

def test_read_sample_sorted_by_date(canned):
    sample = rw.read_waterdata_sample("01491000", "00631")
    assert sample["Date"].is_monotonic_increasing

def test_read_sample_empty_response_raises(canned):
    canned["_samples_response"] = pd.DataFrame()
    with pytest.raises(ValueError, match="No samples"):
        rw.read_waterdata_sample("01491000", "00631")


# ---------------------------------------------------------------------------
# Site metadata
# ---------------------------------------------------------------------------

# --- ReadInfo ----------------------------------------------------------

def test_read_info_maps_name_and_drainage_area(stub_waterdata):
    stub_waterdata["_info_response"] = pd.DataFrame([{
        "id": "USGS-01491000",
        "monitoring_location_name": "CHOPTANK RIVER NEAR GREENSBORO, MD",
        "drainage_area": 113.0,
    }])
    info = rw.read_waterdata_info("01491000", "00631")
    assert info["shortName"].iloc[0] == "CHOPTANK RIVER NEAR GREENSBORO, MD"
    # 113 square miles is EGRET's 292.67 square kilometres
    assert info["drainSqKm"].iloc[0] == pytest.approx(292.67, abs=0.1)
    assert info["paramShortName"].iloc[0] == "Nitrate plus nitrite"

def test_read_info_missing_area_is_nan_not_an_error(stub_waterdata):
    stub_waterdata["_info_response"] = pd.DataFrame([{
        "id": "USGS-01491000",
        "monitoring_location_name": "Somewhere",
    }])
    info = rw.read_waterdata_info("01491000")
    assert np.isnan(info["drainSqKm"].iloc[0])

def test_read_info_empty_response_raises(stub_waterdata):
    stub_waterdata["_info_response"] = pd.DataFrame()
    with pytest.raises(ValueError, match="No monitoring location"):
        rw.read_waterdata_info("01491000")


def test_read_info_takes_parameter_name_and_units_from_the_code_table(stub_waterdata):
    """As R readNWISInfo does, from the USGS parameter-code table."""
    stub_waterdata["_info_response"] = pd.DataFrame([{
        "id": "USGS-01491000",
        "monitoring_location_name": "CHOPTANK RIVER NEAR GREENSBORO, MD",
        "drainage_area": 113.0,
    }])
    stub_waterdata["_reference_response"] = pd.DataFrame([{
        "parameter_code": "00631", "parameter_name": "NO3+NO2, wf",
        "unit_of_measure": "mg/l as N",
    }])
    info = rw.read_waterdata_info("01491000", "00631")
    assert stub_waterdata["reference"] == ("parameter-codes", {"id": "00631"})
    assert info["paramShortName"].iloc[0] == "NO3+NO2, wf"
    assert info["param.units"].iloc[0] == "mg/l as N"
    assert info["paramNumber"].iloc[0] == "00631"
    assert info["station_nm"].iloc[0] == "CHOPTANK RIVER NEAR GREENSBORO, MD"
    assert info["paStart"].iloc[0] == 10
    assert info["paLong"].iloc[0] == 12


def test_read_info_warns_when_units_are_not_mg_per_litre(stub_waterdata):
    stub_waterdata["_info_response"] = pd.DataFrame([{
        "id": "USGS-01491000", "monitoring_location_name": "Somewhere",
    }])
    stub_waterdata["_reference_response"] = pd.DataFrame([{
        "parameter_code": "00095", "parameter_name": "Sp cond",
        "unit_of_measure": "uS/cm @25C",
    }])
    with pytest.warns(UserWarning, match="mg/L"):
        rw.read_waterdata_info("01491000", "00095")

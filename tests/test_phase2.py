"""Phase 2 tests: data population, user readers, merge_report, compress_data."""
import numpy as np
import pandas as pd
import pytest

from pyegret.data.populate import (
    populate_date_columns,
    populate_sample_columns,
    merge_daily_q,
)
from pyegret.data.read_user import read_user_daily, read_user_sample
from pyegret.data.merge_report import merge_report
from pyegret.data.compress_data import compress_data, remove_duplicates, remove_zeros
from pyegret.elist import has_sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_daily():
    """Minimal valid Daily DataFrame (post-populate_date_columns)."""
    dates = pd.date_range("2000-01-01", periods=365, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Date": dates, "Q": rng.uniform(1, 100, 365)})
    df["Q"] = df["Q"].clip(lower=0.01)
    df["LogQ"] = np.log(df["Q"])
    df = populate_date_columns(df)
    return df


@pytest.fixture
def simple_info():
    return pd.DataFrame({
        "station_nm": ["Test River"],
        "site_no": ["12345678"],
        "shortName": ["Test River"],
        "paramShortName": ["Nitrate"],
        "param_cd": ["00631"],
        "lat_va": [39.0],
        "long_va": [-76.0],
        "drain_area_va": [100.0],
    })


@pytest.fixture
def simple_sample(simple_daily):
    """10 synthetic WQ samples within the Daily date range."""
    np.random.seed(42)
    dates = pd.to_datetime([
        "2000-02-10", "2000-03-15", "2000-04-20", "2000-05-05",
        "2000-06-01", "2000-07-07", "2000-08-12", "2000-09-03",
        "2000-10-18", "2000-11-22",
    ])
    conc = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8, 0.4, 1.1])
    uncen = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1])
    df = pd.DataFrame({
        "Date": dates,
        "ConcHigh": conc,
        "ConcLow": np.where(uncen == 0, 0.0, conc),
        "Uncen": uncen,
    })
    df = populate_date_columns(df)
    df = populate_sample_columns(df)
    return df


# ---------------------------------------------------------------------------
# populate_date_columns
# ---------------------------------------------------------------------------

def test_populate_date_columns_adds_all_cols():
    df = pd.DataFrame({"Date": pd.to_datetime(["2000-06-15"])})
    df = populate_date_columns(df)
    for col in ("DecYear", "Month", "Day", "MonthSeq", "Julian", "SinDY", "CosDY"):
        assert col in df.columns, f"Missing: {col}"


def test_populate_date_columns_dec_year():
    df = pd.DataFrame({"Date": pd.to_datetime(["2000-06-15"])})
    df = populate_date_columns(df)
    # 2000-06-15 is day 167 of leap year 366 → (167-1)/366 ≈ 0.4536
    assert abs(df["DecYear"].iloc[0] - 2000.45) < 0.01


def test_populate_date_columns_sin_cos():
    df = pd.DataFrame({"Date": pd.to_datetime(["2000-01-01"])})
    df = populate_date_columns(df)
    # DecYear ≈ 2000.0 → sin(2π×2000) = 0, cos = 1
    assert abs(df["SinDY"].iloc[0]) < 0.01
    assert abs(df["CosDY"].iloc[0] - 1.0) < 0.01


def test_populate_date_columns_month_and_day():
    df = pd.DataFrame({"Date": pd.to_datetime(["2001-03-05"])})
    df = populate_date_columns(df)
    assert df["Month"].iloc[0] == 3
    # 2001 is a common year, so EGRET shifts days after Feb 28 up by one:
    # plain day-of-year 64 becomes 65.
    assert df["Day"].iloc[0] == 65


def test_populate_date_columns_day_is_leap_aligned():
    """March 1 must be day 61 in both a leap and a common year."""
    df = pd.DataFrame({"Date": pd.to_datetime(["2000-03-01", "2001-03-01"])})
    df = populate_date_columns(df)
    assert df["Day"].tolist() == [61, 61]


def test_populate_date_columns_leap_day_is_60():
    df = pd.DataFrame({"Date": pd.to_datetime(["2000-02-29"])})
    df = populate_date_columns(df)
    assert df["Day"].iloc[0] == 60


def test_populate_date_columns_before_march_is_unshifted():
    df = pd.DataFrame({"Date": pd.to_datetime(["2001-02-28", "2001-01-01"])})
    df = populate_date_columns(df)
    assert df["Day"].tolist() == [59, 1]


def test_populate_date_columns_adds_water_year():
    df = pd.DataFrame({"Date": pd.to_datetime(["1999-10-01", "2000-09-30"])})
    df = populate_date_columns(df)
    assert df["waterYear"].tolist() == [2000, 2000]


def test_populate_date_columns_preserves_existing_cols():
    df = pd.DataFrame({"Date": pd.to_datetime(["2000-01-01"]), "Q": [1.5]})
    df = populate_date_columns(df)
    assert "Q" in df.columns
    assert df["Q"].iloc[0] == 1.5


# ---------------------------------------------------------------------------
# populate_sample_columns
# ---------------------------------------------------------------------------

def test_populate_sample_columns_conc_ave():
    df = pd.DataFrame({"ConcLow": [0.0, 1.0], "ConcHigh": [2.0, 2.0], "Uncen": [0, 1]})
    df = populate_sample_columns(df)
    assert df["ConcAve"].tolist() == [1.0, 1.5]


def test_populate_sample_columns_log_high():
    df = pd.DataFrame({"ConcLow": [1.0], "ConcHigh": [np.e], "Uncen": [1]})
    df = populate_sample_columns(df)
    assert abs(df["LogConcHigh"].iloc[0] - 1.0) < 1e-9


def test_populate_sample_columns_conc_ave_censored_zero_lower():
    """PyEGRET spells a censored lower bound as 0."""
    df = pd.DataFrame({"ConcLow": [0.0], "ConcHigh": [0.05], "Uncen": [0]})
    out = populate_sample_columns(df)
    assert out["ConcAve"].iloc[0] == pytest.approx(0.025)


def test_populate_sample_columns_conc_ave_censored_na_lower():
    """R EGRET spells the same thing as NA; both must give ConcHigh / 2."""
    df = pd.DataFrame({"ConcLow": [np.nan], "ConcHigh": [0.05], "Uncen": [0]})
    out = populate_sample_columns(df)
    assert out["ConcAve"].iloc[0] == pytest.approx(0.025)


def test_populate_sample_columns_log_low_nan_when_zero():
    """ConcLow = 0 (censored) → LogConcLow must be NaN."""
    df = pd.DataFrame({"ConcLow": [0.0], "ConcHigh": [0.5], "Uncen": [0]})
    df = populate_sample_columns(df)
    assert np.isnan(df["LogConcLow"].iloc[0])


def test_populate_sample_columns_log_low_positive():
    df = pd.DataFrame({"ConcLow": [np.e], "ConcHigh": [np.e], "Uncen": [1]})
    df = populate_sample_columns(df)
    assert abs(df["LogConcLow"].iloc[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# merge_daily_q
# ---------------------------------------------------------------------------

def test_populate_concentrations_merges_q(simple_daily, simple_sample):
    merged = merge_daily_q(simple_daily, simple_sample)
    # After merge, Q and LogQ should be present and not all NaN
    assert "Q" in merged.columns
    assert "LogQ" in merged.columns
    assert np.any(merged["Q"].notna())


def test_populate_concentrations_q_positive(simple_daily, simple_sample):
    merged = merge_daily_q(simple_daily, simple_sample)
    assert np.all((merged["Q"].dropna() > 0))


def test_populate_concentrations_preserves_sample_rows(simple_daily, simple_sample):
    merged = merge_daily_q(simple_daily, simple_sample)
    assert len(merged) == len(simple_sample)


# ---------------------------------------------------------------------------
# read_user_daily
# ---------------------------------------------------------------------------

def test_read_user_daily_basic(tmp_path):
    csv = tmp_path / "daily.csv"
    csv.write_text("date,discharge\n2000-01-01,1.5\n2000-01-02,2.0\n")
    df = read_user_daily(str(csv), q_unit=2, verbose=False)
    assert list(df["Q"]) == [1.5, 2.0]
    assert "LogQ" in df.columns
    assert "DecYear" in df.columns


def test_read_user_daily_assumes_cfs_like_egret(tmp_path):
    # R readUserDaily defaults to qUnit = 1: the file is in ft3/s.
    csv = tmp_path / "daily.csv"
    csv.write_text("date,discharge\n2000-01-01,35.314667\n")
    df = read_user_daily(str(csv), verbose=False)
    assert df["Q"].iloc[0] == pytest.approx(1.0)


def test_read_user_daily_shifts_zero_flows_like_egret(tmp_path):
    # populateDaily adds 0.001 * mean(Q) to every day when there are zero
    # flows and no negative ones, so log(Q) is defined everywhere.
    csv = tmp_path / "daily.csv"
    csv.write_text("date,discharge\n2000-01-01,0.0\n2000-01-02,2.0\n2000-01-03,4.0\n")
    df = read_user_daily(str(csv), q_unit=2, verbose=False)
    assert len(df) == 3
    shift = 0.001 * 2.0
    assert df["Q"].tolist() == pytest.approx([shift, 2.0 + shift, 4.0 + shift])


def test_read_user_daily_fills_the_calendar(tmp_path):
    # Missing days are kept as NaN by populateDaily, then readUserDaily drops
    # them, so the result has only the reported days.
    csv = tmp_path / "daily.csv"
    csv.write_text("date,discharge\n2000-01-01,1.0\n2000-01-04,2.0\n")
    df = read_user_daily(str(csv), q_unit=2, verbose=False)
    assert len(df) == 2


def test_read_user_daily_us_dates(tmp_path):
    csv = tmp_path / "daily.csv"
    csv.write_text("date,discharge\n03/15/2001,1.0\n03/16/2001,2.0\n")
    df = read_user_daily(str(csv), q_unit=2, verbose=False)
    assert df["Date"].iloc[0] == pd.Timestamp("2001-03-15")


def test_read_user_daily_logq_correct(tmp_path):
    csv = tmp_path / "daily.csv"
    csv.write_text("date,discharge\n2000-06-01,1.0\n")
    df = read_user_daily(str(csv), q_unit=2, verbose=False)
    assert abs(df["LogQ"].iloc[0] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# read_user_sample
# ---------------------------------------------------------------------------

def test_read_user_sample_uncen_flags(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,remark,value\n2000-01-15,,1.4\n2000-02-10,<,0.5\n")
    df = read_user_sample(str(csv))
    assert df["Uncen"].tolist() == [1, 0]


def test_read_user_sample_conc_low_censored(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,remark,value\n2000-01-15,,1.4\n2000-02-10,<,0.5\n")
    df = read_user_sample(str(csv))
    assert df["ConcLow"].iloc[1] == 0.0


def test_read_user_sample_detected_conclow_equals_conchigh(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,remark,value\n2000-01-15,,1.4\n")
    df = read_user_sample(str(csv))
    assert df["ConcLow"].iloc[0] == df["ConcHigh"].iloc[0]


def test_read_user_sample_has_log_cols(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,remark,value\n2000-01-15,,1.4\n")
    df = read_user_sample(str(csv))
    assert "LogConcHigh" in df.columns
    assert "ConcAve" in df.columns


def test_read_user_sample_date_columns_added(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text("date,remark,value\n2000-06-15,,0.9\n")
    df = read_user_sample(str(csv))
    assert "DecYear" in df.columns
    assert "SinDY" in df.columns


# ---------------------------------------------------------------------------
# merge_report
# ---------------------------------------------------------------------------

def test_merge_report_returns_elist(simple_info, simple_daily):
    elist = merge_report(simple_info, simple_daily)
    assert isinstance(elist, dict)


def test_merge_report_no_sample(simple_info, simple_daily):
    elist = merge_report(simple_info, simple_daily)
    assert not has_sample(elist)


def test_merge_report_with_sample(simple_info, simple_daily, simple_sample):
    elist = merge_report(simple_info, simple_daily, simple_sample)
    assert has_sample(elist)
    assert "Q" in elist["Sample"].columns


def test_merge_report_missing_col_raises(simple_info):
    bad_daily = pd.DataFrame({"Date": pd.to_datetime(["2000-01-01"]), "Q": [1.0]})
    with pytest.raises(ValueError, match="missing required column"):
        merge_report(simple_info, bad_daily)


def test_merge_report_warns_sample_before_daily(simple_info, simple_daily):
    early_sample = pd.DataFrame({
        "Date": pd.to_datetime(["1999-01-01"]),
        "ConcHigh": [1.0],
        "ConcLow": [1.0],
        "Uncen": [1],
    })
    early_sample = populate_date_columns(early_sample)
    early_sample = populate_sample_columns(early_sample)
    with pytest.warns(UserWarning, match="before the start"):
        merge_report(simple_info, simple_daily, early_sample)


def test_merge_report_warns_sample_after_daily(simple_info, simple_daily):
    late_sample = pd.DataFrame({
        "Date": pd.to_datetime(["2050-01-01"]),
        "ConcHigh": [1.0],
        "ConcLow": [1.0],
        "Uncen": [1],
    })
    late_sample = populate_date_columns(late_sample)
    late_sample = populate_sample_columns(late_sample)
    with pytest.warns(UserWarning, match="after the end"):
        merge_report(simple_info, simple_daily, late_sample)


# ---------------------------------------------------------------------------
# compress_data / remove_zeros: R compressData semantics
# ---------------------------------------------------------------------------

def _raw(rows, columns=("date", "rn", "nitrate", "rd", "nitrite")):
    return pd.DataFrame(rows, columns=list(columns))


def test_compress_data_single_pair():
    raw = _raw([["2000-01-01", "", "1.4"], ["2000-02-01", "<", "0.5"]],
               columns=("date", "remark", "value"))
    out = compress_data(raw)
    assert out["ConcLow"].tolist() == [1.4, 0.0]
    assert out["ConcHigh"].tolist() == [1.4, 0.5]
    assert out["Uncen"].tolist() == [1, 0]


def test_compress_data_sums_the_pairs():
    raw = _raw([["2001-03-01", "", "1.2", "<", "0.01"]])
    out = compress_data(raw)
    assert out["ConcLow"].iloc[0] == pytest.approx(1.2)
    assert out["ConcHigh"].iloc[0] == pytest.approx(1.21)
    assert out["Uncen"].iloc[0] == 0


def test_compress_data_both_detected_is_uncensored():
    raw = _raw([["2001-03-01", "", "1.2", "", "0.3"]])
    out = compress_data(raw)
    assert out["ConcLow"].iloc[0] == pytest.approx(1.5)
    assert out["Uncen"].iloc[0] == 1


def test_compress_data_greater_than_has_no_upper_bound():
    raw = _raw([["2001-03-01", ">", "50"]], columns=("date", "remark", "value"))
    out = compress_data(raw)
    assert out["ConcLow"].iloc[0] == 50.0
    assert out["ConcHigh"].iloc[0] == 0.0
    assert out["Uncen"].iloc[0] == 0


def test_remove_zeros_drops_zero_results():
    raw = _raw([["2000-01-01", "", "0.0"], ["2000-02-01", "", "1.0"]],
               columns=("date", "remark", "value"))
    out = compress_data(raw)
    with pytest.warns(UserWarning, match="reported as 0.0"):
        kept = remove_zeros(out, verbose=False)
    assert len(kept) == 1


def test_read_user_sample_right_censored_conc_ave(tmp_path):
    # populateSampleColumns: ConcHigh becomes NA and ConcAve = 1.5 * ConcLow.
    csv = tmp_path / "sample.csv"
    csv.write_text("date,remark,value\n2000-01-15,>,50\n")
    df = read_user_sample(str(csv))
    assert np.isnan(df["ConcHigh"].iloc[0])
    assert df["ConcAve"].iloc[0] == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

def test_remove_duplicates_drops_exact():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2000-01-01", "2000-01-01"]),
        "ConcHigh": [1.0, 1.0],
        "Uncen": [1, 1],
    })
    out = remove_duplicates(df)
    assert len(out) == 1


def test_remove_duplicates_keeps_different_conc():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2000-01-01", "2000-01-01"]),
        "ConcHigh": [1.0, 2.0],
        "Uncen": [1, 1],
    })
    out = remove_duplicates(df)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Network tests (skipped unless --run-network is passed)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_read_nwis_daily_choptank():
    from pyegret.data.read_nwis import read_nwis_daily
    df = read_nwis_daily("01491000", "00060", start_date="2000-01-01", end_date="2000-12-31")
    assert "Q" in df.columns
    assert len(df) > 0
    assert np.all((df["Q"] > 0))


@pytest.mark.network
def test_read_nwis_sample_choptank():
    from pyegret.data.read_nwis import read_nwis_sample
    with pytest.warns(FutureWarning, match="read_waterdata_sample"):
        df = read_nwis_sample("01491000", "00631", start_date="2000-01-01",
                              end_date="2005-12-31")
    assert "ConcHigh" in df.columns
    assert len(df) > 0


@pytest.mark.network
def test_read_nwis_info_choptank():
    from pyegret.data.read_nwis import read_nwis_info
    info = read_nwis_info("01491000", "00631")
    assert "shortName" in info.columns
    assert "param_cd" in info.columns
    assert info["param_cd"].iloc[0] == "00631"

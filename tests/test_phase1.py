"""Phase 1 tests: utils and eList helpers."""
import pytest
import pandas as pd
import numpy as np

from pyegret.utils.date_utils import (
    to_decimal_year,
    month_seq,
    julian_day,
    check_date_format,
)
from pyegret.utils.units import get_q_unit, get_flux_unit
from pyegret.utils.season import set_season_label
from pyegret.elist import make_elist, has_sample, has_model, describe_elist


# ---------------------------------------------------------------------------
# to_decimal_year
# ---------------------------------------------------------------------------

def test_decimal_year_jan1():
    dates = pd.to_datetime(["1985-01-01", "1985-07-02", "2000-01-01"])
    dy = to_decimal_year(dates)
    assert abs(dy.iloc[0] - 1985.0) < 0.01


def test_decimal_year_mid_year():
    dates = pd.to_datetime(["1985-01-01", "1985-07-02", "2000-01-01"])
    dy = to_decimal_year(dates)
    assert abs(dy.iloc[1] - 1985.5) < 0.01


def test_decimal_year_new_year_2000():
    dates = pd.to_datetime(["2000-01-01"])
    dy = to_decimal_year(dates)
    assert abs(dy.iloc[0] - 2000.0) < 0.001


def test_decimal_year_leap_year():
    # 2000 is a leap year; day 183 / 366
    dates = pd.to_datetime(["2000-07-01"])
    dy = to_decimal_year(dates)
    # day_of_year for 2000-07-01 = 183; (183-1)/366 = 0.4973
    expected = 2000 + 182 / 366
    assert abs(dy.iloc[0] - expected) < 0.001


# ---------------------------------------------------------------------------
# month_seq
# ---------------------------------------------------------------------------

def test_month_seq_is_absolute_from_1850():
    """EGRET's MonthSeq is (year - 1850) * 12 + month, not record-relative."""
    dates = pd.to_datetime(["1850-01-15", "1850-02-10", "2000-03-20"])
    seq = month_seq(dates)
    assert seq.tolist() == [1, 2, (2000 - 1850) * 12 + 3]


def test_month_seq_consecutive_months_differ_by_one():
    dates = pd.to_datetime(["2000-01-15", "2000-02-10", "2000-03-20"])
    seq = month_seq(dates)
    assert list(np.diff(seq.to_numpy())) == [1, 1]


def test_month_seq_explicit_base():
    dates = pd.to_datetime(["2001-01-01", "2001-03-01"])
    base = pd.Timestamp("2000-10-01")
    seq = month_seq(dates, base_date=base)
    # Oct=1, Nov=2, Dec=3, Jan=4, Feb=5, Mar=6
    assert seq.tolist() == [4, 6]


def test_month_seq_returns_int():
    dates = pd.to_datetime(["2000-06-15"])
    seq = month_seq(dates)
    assert seq.dtype == np.int64


# ---------------------------------------------------------------------------
# julian_day
# ---------------------------------------------------------------------------

def test_julian_day_origin():
    dates = pd.to_datetime(["1850-01-01"])
    jd = julian_day(dates)
    assert jd.iloc[0] == 0


def test_julian_day_next_day():
    dates = pd.to_datetime(["1850-01-02"])
    jd = julian_day(dates)
    assert jd.iloc[0] == 1


def test_julian_day_returns_int():
    dates = pd.to_datetime(["2000-01-01"])
    jd = julian_day(dates)
    assert jd.dtype == np.int64


# ---------------------------------------------------------------------------
# check_date_format
# ---------------------------------------------------------------------------

def test_check_date_format_empty():
    assert check_date_format("") is True


def test_check_date_format_valid():
    assert check_date_format("2000-01-15") is True
    assert check_date_format("1985-12-31") is True


def test_check_date_format_invalid_separator():
    assert check_date_format("2000/01/15") is False


def test_check_date_format_invalid_string():
    assert check_date_format("not-a-date") is False


def test_check_date_format_partial():
    assert check_date_format("2000-01") is False


# ---------------------------------------------------------------------------
# get_q_unit / get_flux_unit: EGRET's own tables and numbering
# ---------------------------------------------------------------------------

def test_q_units_cfs():
    u = get_q_unit(1)
    assert u["name"] == "cfs"
    assert u["factor"] == pytest.approx(35.314667)


def test_q_units_cms():
    u = get_q_unit(2)
    assert u["name"] == "cms"
    assert u["factor"] == 1.0


def test_q_units_by_name_matches_number():
    assert get_q_unit("thousandCfs") is get_q_unit(3)


def test_q_units_all_keys():
    for k in range(1, 6):
        u = get_q_unit(k)
        assert "name" in u and "factor" in u and "label" in u


def test_q_units_invalid_key():
    with pytest.raises(KeyError):
        get_q_unit(99)


def test_flux_unit_numbering_follows_egret():
    # EGRET's fluxUnit = 1 is pounds/day and 3 is kg/day; a ported R script
    # with fluxUnit = 3 must get kg/day.
    assert get_flux_unit(1)["name"] == "poundsDay"
    assert get_flux_unit(3)["name"] == "kgDay"
    assert get_flux_unit(3)["factor"] == 1.0
    assert get_flux_unit(13)["name"] == "kgYear"


def test_flux_unit_year_factors_use_365_25_days():
    assert get_flux_unit("kgYear")["factor"] == pytest.approx(365.25)
    assert get_flux_unit("thousandKgYear")["factor"] == pytest.approx(0.36525)
    assert get_flux_unit("tonsYear")["factor"] == pytest.approx(0.402619)


def test_flux_units_all_keys():
    for k in range(1, 14):
        u = get_flux_unit(k)
        assert "name" in u and "factor" in u and "label" in u


def test_flux_units_invalid_key():
    with pytest.raises(KeyError):
        get_flux_unit(0)
    with pytest.raises(KeyError):
        get_flux_unit("kgPerFortnight")


# ---------------------------------------------------------------------------
# set_season_label: EGRET's wording
# ---------------------------------------------------------------------------

def test_season_label_water_year():
    assert set_season_label(10, 12) == "Water Year"


def test_season_label_calendar_year():
    assert set_season_label(1, 12) == "Calendar Year"


def test_season_label_other_twelve_months():
    assert set_season_label(4, 12) == "Year Starting With April"


def test_season_label_custom():
    assert set_season_label(4, 6) == "Season Consisting of Apr May Jun Jul Aug Sep"


def test_season_label_single_month():
    assert set_season_label(7, 1) == "Season Consisting of Jul"


def test_season_label_wrap_around():
    assert set_season_label(11, 3) == "Season Consisting of Nov Dec Jan"


# ---------------------------------------------------------------------------
# eList
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_elist():
    info = pd.DataFrame({
        "station_nm": ["Test River"],
        "site_no": ["12345678"],
    })
    daily = pd.DataFrame({
        "Date": pd.to_datetime(["2000-01-01", "2000-01-02"]),
        "Q": [1.0, 1.5],
    })
    return make_elist(INFO=info, Daily=daily)


def test_elist_has_model_false(simple_elist):
    assert has_model(simple_elist) is False


def test_elist_has_sample_false(simple_elist):
    assert has_sample(simple_elist) is False


def test_elist_has_model_true(simple_elist):
    simple_elist["surfaces"] = np.zeros((16, 14, 3))
    assert has_model(simple_elist) is True


def test_elist_has_sample_true(simple_elist):
    simple_elist["Sample"] = pd.DataFrame({"ConcHigh": [1.0]})
    assert has_sample(simple_elist) is True


def test_describe_elist_contains_site(simple_elist):
    r = describe_elist(simple_elist)
    assert "Test River" in r


def test_describe_elist_contains_dates(simple_elist):
    r = describe_elist(simple_elist)
    assert "2000-01-01" in r
    assert "2000-01-02" in r


def test_describe_elist_model_no(simple_elist):
    assert "Model:   no" in describe_elist(simple_elist)


# ---------------------------------------------------------------------------
# top-level pyegret import
# ---------------------------------------------------------------------------

def test_egret_import():
    import pyegret
    assert pyegret.__version__ == "0.2.0"
    assert pyegret.make_elist is make_elist

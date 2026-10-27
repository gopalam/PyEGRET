"""
Behaviours that must match R EGRET exactly, each checked against what the R
source in reference/EGRET_R does.
"""
import numpy as np
import pandas as pd
import pytest

from pyegret.data.populate import merge_daily_q, populate_date_columns, populate_sample_columns
from pyegret.model.run_surv_reg import run_surv_reg


def _regression_sample(n=300, seed=3):
    """A sample with a known log-linear truth and no censoring."""
    rng = np.random.default_rng(seed)
    dec_year = rng.uniform(1995, 2005, n)
    log_q = rng.normal(0.0, 1.0, n)
    log_c = 0.2 - 0.03 * (dec_year - 2000) + 0.4 * log_q + 0.3 * np.sin(2 * np.pi * dec_year)
    log_c = log_c + rng.normal(0.0, 0.25, n)
    sample = pd.DataFrame({
        "DecYear": dec_year,
        "LogQ": log_q,
        "SinDY": np.sin(2 * np.pi * dec_year),
        "CosDY": np.cos(2 * np.pi * dec_year),
        "ConcLow": np.exp(log_c),
        "ConcHigh": np.exp(log_c),
        "Uncen": np.ones(n, dtype=int),
    })
    sample["LogConcHigh"] = np.log(sample["ConcHigh"])
    return sample


def _fit(sample):
    weights = np.ones(len(sample))
    return run_surv_reg(2000.0, 0.0, 0.0, 1.0, sample, weights)


# ---------------------------------------------------------------------------
# runSurvReg: Surv(log(ConcLow), log(ConcHigh), type = "interval2")
# ---------------------------------------------------------------------------

def test_left_censoring_same_from_bounds_or_from_uncen():
    sample = _regression_sample()
    censored = sample["ConcHigh"] < np.quantile(sample["ConcHigh"], 0.2)
    sample.loc[censored, "ConcLow"] = 0.0
    sample.loc[censored, "Uncen"] = 0
    from_bounds = _fit(sample)
    legacy = sample.drop(columns=["ConcLow", "ConcHigh"])
    from_uncen = _fit(legacy)
    assert from_bounds[0] == pytest.approx(from_uncen[0], abs=1e-6)
    assert from_bounds[1] == pytest.approx(from_uncen[1], abs=1e-6)


def test_interval_censoring_uses_the_lower_bound():
    # Turning detections into narrow intervals around the true value should
    # barely move the fit; treating the same rows as left-censored at their
    # upper bound should move it a lot more.
    sample = _regression_sample()
    exact = _fit(sample)

    rows = np.arange(0, len(sample), 3)
    interval = sample.copy()
    interval.loc[rows, "ConcLow"] = sample.loc[rows, "ConcHigh"] * 0.9
    interval.loc[rows, "ConcHigh"] = sample.loc[rows, "ConcHigh"] * 1.1
    interval.loc[rows, "Uncen"] = 0

    left = interval.copy()
    left.loc[rows, "ConcLow"] = 0.0

    fit_interval = _fit(interval)
    fit_left = _fit(left)
    assert abs(fit_interval[0] - exact[0]) < 0.01
    assert abs(fit_left[0] - exact[0]) > 3 * abs(fit_interval[0] - exact[0])


def test_right_censored_value_has_no_upper_bound():
    # A ">" result: ConcLow is the limit, ConcHigh is missing. It must pull
    # the fit up, not be read as a value of zero.
    sample = _regression_sample()
    top = sample["ConcHigh"] > np.quantile(sample["ConcHigh"], 0.9)
    right = sample.copy()
    right.loc[top, "ConcLow"] = sample.loc[top, "ConcHigh"] * 0.8
    right.loc[top, "ConcHigh"] = np.nan
    right.loc[top, "Uncen"] = 0
    exact = _fit(sample)
    fit_right = _fit(right)
    assert np.isfinite(fit_right[0])
    assert abs(fit_right[0] - exact[0]) < 0.05


# ---------------------------------------------------------------------------
# mergeReport: merge(..., by = "Date", all.y = TRUE)
# ---------------------------------------------------------------------------

def test_merge_matches_dates_exactly():
    daily = pd.DataFrame({
        "Date": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"]),
        "Q": [1.0, 2.0, 4.0],
    })
    daily["LogQ"] = np.log(daily["Q"])
    sample = pd.DataFrame({"Date": pd.to_datetime(["2000-01-02", "2000-01-03"])})
    with pytest.warns(UserWarning, match="do not have corresponding flow data"):
        merged = merge_daily_q(daily, sample)
    assert merged["Q"].iloc[0] == 2.0
    # 2000-01-03 has no flow: R keeps the sample with Q = NA rather than
    # borrowing a neighbouring day's discharge.
    assert np.isnan(merged["Q"].iloc[1])


# ---------------------------------------------------------------------------
# populateSampleColumns: ConcAve
# ---------------------------------------------------------------------------

def test_conc_ave_for_each_kind_of_sample():
    sample = pd.DataFrame({
        "Date": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
        "ConcLow": [1.0, 0.0, 50.0],
        "ConcHigh": [1.0, 0.05, 0.0],
        "Uncen": [1, 0, 0],
    })
    sample = populate_date_columns(sample)
    sample = populate_sample_columns(sample)
    assert sample["ConcAve"].tolist()[:2] == pytest.approx([1.0, 0.025])
    assert sample["ConcAve"].iloc[2] == pytest.approx(75.0)
    assert np.isnan(sample["ConcHigh"].iloc[2])


# ---------------------------------------------------------------------------
# Residuals: log(ConcAve) - yHat
# ---------------------------------------------------------------------------

def test_cross_validated_resid_uses_conc_ave():
    from pyegret.testing import make_synthetic_elist
    from pyegret.model.est_cross_val import est_cross_val
    elist = make_synthetic_elist(n_samples=120, censor_fraction=0.2, seed=4)
    sample = est_cross_val(elist, verbose=False)
    expected = np.log(sample["ConcAve"].to_numpy()) - sample["yHat"].to_numpy()
    assert np.allclose(sample["resid"].to_numpy(), expected, equal_nan=True)

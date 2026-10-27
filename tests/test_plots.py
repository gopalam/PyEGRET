"""
Every EGRET plot draws without error and returns (fig, ax) with content.

These are smoke tests; they check what is on the axes, not how it looks.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pyegret.plots as P  # noqa: E402
from pyegret.model.kalman import wrtds_kalman  # noqa: E402
from pyegret.model.model_estimation import model_estimation  # noqa: E402
from pyegret.plots.plot_utils import log_pretty3  # noqa: E402
from pyegret.testing import make_synthetic_elist  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    elist = make_synthetic_elist(start="1998-01-01", end="2005-12-31", n_samples=200,
                                 censor_fraction=0.15, seed=7)
    elist["INFO"]["drainSqKm"] = 300.0
    return elist


@pytest.fixture(scope="module")
def fitted(raw):
    elist = model_estimation(raw, verbose=False)
    return wrtds_kalman(elist, niter=10, seed=1, verbose=False)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _has_data(ax):
    return len(ax.lines) + len(ax.collections) + len(ax.patches) > 0


@pytest.mark.parametrize("name, kwargs", [
    ("plot_conc_q", {}),
    ("plot_conc_time", {}),
    ("plot_conc_time", {"q_lower": 0.5, "q_upper": 5}),
    ("box_conc_month", {}),
    ("box_q_twice", {}),
    ("plot_q_time_daily", {}),
    ("plot_q_time_daily", {"q_lower": 30}),
])
def test_data_plots(raw, name, kwargs):
    fig, ax = getattr(P, name)(raw, **kwargs)
    assert _has_data(ax)
    assert ax.get_title() != ""


def test_multi_plot_data_overview(raw):
    fig, axes = P.multi_plot_data_overview(raw)
    assert axes.shape == (2, 2)
    for ax in axes.ravel():
        assert _has_data(ax)


@pytest.mark.parametrize("name, kwargs", [
    ("plot_conc_hist", {}),
    ("plot_conc_hist", {"plot_gen_conc": True}),
    ("plot_flux_hist", {"flux_unit": 3}),
    ("plot_conc_time_daily", {"year_start": 2000, "year_end": 2002}),
    ("plot_conc_time_daily", {"random_censored": True}),
    ("plot_flux_time_daily", {}),
    ("plot_resid_pred", {}),
    ("plot_resid_pred", {"std_resid": True, "random_censored": True}),
    ("plot_resid_q", {}),
    ("plot_resid_time", {}),
    ("box_resid_month", {}),
    ("box_conc_three", {}),
    ("plot_conc_pred", {}),
    ("plot_conc_pred", {"log_scale": True}),
    ("plot_flux_pred", {}),
])
def test_model_plots(fitted, name, kwargs):
    fig, ax = getattr(P, name)(fitted, **kwargs)
    assert _has_data(ax)


def test_conc_hist_draws_the_flow_normalized_line(fitted):
    fig, ax = P.plot_conc_hist(fitted)
    from pyegret.analysis.setup_years import setup_years
    annual = setup_years(fitted["Daily"])
    line = [ln for ln in ax.lines if ln.get_linestyle() == "-"][0]
    assert np.allclose(line.get_ydata(), annual["FNConc"].to_numpy())


def test_flux_bias_multi(fitted):
    fig, axes = P.flux_bias_multi(fitted)
    assert axes.shape == (4, 2)
    assert "Flux Bias Statistic" in fig._suptitle.get_text()


def test_plot_wrtds_kalman(fitted):
    fig, axes = P.plot_wrtds_kalman(fitted)
    assert len(axes) == 2


def test_plot_wrtds_kalman_needs_kalman(raw):
    with pytest.raises(ValueError, match="wrtds_kalman"):
        P.plot_wrtds_kalman(raw)


def test_plot_contours(fitted):
    fig, ax = P.plot_contours(fitted, 1999, 2005)
    assert len(ax.collections) > 0
    assert "flow percentiles" in ax.get_title()


def test_plot_diff_contours(fitted):
    fig, ax = P.plot_diff_contours(fitted, 1999, 2004, plot_percent=True)
    assert "percent change" in ax.get_title()


def test_plot_conc_q_smooth(fitted):
    fig, ax = P.plot_conc_q_smooth(fitted, "1999-06-01", "2004-06-01", q_low=0.3, q_high=5)
    assert len(ax.lines) == 2


def test_plot_conc_time_smooth(fitted):
    fig, ax = P.plot_conc_time_smooth(fitted, 0.5, 2.0, center_date="06-01")
    assert len(ax.lines) == 2


def test_log_pretty3_matches_r_examples():
    # From the R documentation: logPretty3(3, 15) and logPretty3(0.7, 990000)
    assert np.allclose(log_pretty3(3, 15), [2, 5, 10, 20])
    ticks = log_pretty3(0.7, 990000)
    assert ticks[0] == pytest.approx(0.5)
    assert ticks[-1] == pytest.approx(1e6)


@pytest.mark.parametrize("conc", [True, False])
def test_plot_time_slice(fitted, conc):
    fig, ax = P.plot_time_slice(fitted, 2001, 2001.5, conc=conc)
    assert "Ratio of means" in ax.get_title()

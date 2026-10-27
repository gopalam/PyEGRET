"""flowHistory: streamflow trend analysis that needs no water-quality data."""
from pyegret.flow.trend_tests import theil_sen_slope, mann_kendall, zhang_prewhitened_test
from pyegret.flow.quantile_kendall import sort_annual_flows, quantile_kendall
from pyegret.flow.plots import plot_quantile_kendall, plot_flow_trend
from pyegret.flow.annual_series import (
    ISTAT_NAMES,
    add_rolling_means,
    make_annual_series,
    flow_duration,
    cum_q_date,
)

__all__ = [
    "ISTAT_NAMES",
    "add_rolling_means",
    "make_annual_series",
    "flow_duration",
    "cum_q_date",
    "theil_sen_slope",
    "mann_kendall",
    "zhang_prewhitened_test",
    "sort_annual_flows",
    "quantile_kendall",
    "plot_quantile_kendall",
    "plot_flow_trend",
]

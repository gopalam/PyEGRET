"""
Post-model analysis: annual aggregation, error statistics, trend comparison.

Streamflow statistics that need no water-quality data live in
:mod:`pyegret.flow` instead - ``make_annual_series``, ``flow_duration``,
``cum_q_date`` and the Quantile-Kendall analysis. Earlier, less faithful
versions of the first two lived here and have been removed rather than left
to shadow them under the same names.
"""
from pyegret.analysis.setup_years import setup_years, set_pa, start_end, get_pa
from pyegret.analysis.monthly_results import calculate_monthly_results
from pyegret.analysis.error_stats import error_stats, flux_bias_stat
from pyegret.analysis.run_pairs import run_pairs, print_pairs
from pyegret.analysis.run_groups import run_groups, print_groups
from pyegret.analysis.run_series import run_series
from pyegret.analysis.trend_setup import surface_start_end
from pyegret.analysis.table_change import table_change, table_change_single, table_results

__all__ = [
    "setup_years",
    "set_pa",
    "start_end",
    "calculate_monthly_results",
    "error_stats",
    "flux_bias_stat",
    "run_pairs",
    "print_pairs",
    "run_groups",
    "print_groups",
    "run_series",
    "surface_start_end",
    "table_change",
    "table_results",
    "table_change_single",
    "get_pa",
]

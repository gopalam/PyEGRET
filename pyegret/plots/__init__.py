"""
EGRET's plots. Every function returns ``(fig, ax)`` - or ``(fig, axes)`` for
multi-panel figures - and never calls ``plt.show()``.

Before a model:   plot_conc_q, plot_conc_time, box_conc_month, box_q_twice,
                  plot_q_time_daily, multi_plot_data_overview
After a model:    plot_conc_hist, plot_flux_hist, plot_conc_time_daily,
                  plot_flux_time_daily, plot_contours, plot_diff_contours,
                  plot_conc_q_smooth, plot_conc_time_smooth
Diagnostics:      plot_resid_pred, plot_resid_q, plot_resid_time,
                  box_resid_month, box_conc_three, plot_conc_pred,
                  plot_flux_pred, flux_bias_multi
WRTDS-Kalman:     plot_wrtds_kalman, plot_time_slice

Streamflow-only figures (flowHistory, Quantile-Kendall) are in pyegret.flow.
"""
from pyegret.plots.data_plots import (
    plot_conc_q,
    plot_conc_time,
    box_conc_month,
    box_q_twice,
    plot_q_time_daily,
    multi_plot_data_overview,
)
from pyegret.plots.history_plots import (
    plot_conc_hist,
    plot_flux_hist,
    plot_conc_time_daily,
    plot_flux_time_daily,
    plot_wrtds_kalman,
    plot_time_slice,
)
from pyegret.plots.diagnostic_plots import (
    plot_resid_pred,
    plot_resid_q,
    plot_resid_time,
    box_resid_month,
    box_conc_three,
    plot_conc_pred,
    plot_flux_pred,
    flux_bias_multi,
)
from pyegret.plots.surface_plots import (
    plot_contours,
    plot_diff_contours,
    plot_conc_q_smooth,
    plot_conc_time_smooth,
)
from pyegret.plots.plot_utils import log_pretty1, log_pretty3, select_days

__all__ = [
    "plot_conc_q", "plot_conc_time", "box_conc_month", "box_q_twice",
    "plot_q_time_daily", "multi_plot_data_overview",
    "plot_conc_hist", "plot_flux_hist", "plot_conc_time_daily",
    "plot_flux_time_daily", "plot_wrtds_kalman", "plot_time_slice",
    "plot_resid_pred", "plot_resid_q", "plot_resid_time", "box_resid_month",
    "box_conc_three", "plot_conc_pred", "plot_flux_pred", "flux_bias_multi",
    "plot_contours", "plot_diff_contours", "plot_conc_q_smooth",
    "plot_conc_time_smooth",
    "log_pretty1", "log_pretty3", "select_days",
]

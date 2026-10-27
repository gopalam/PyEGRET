# Changelog

## 0.1.0 - 2026-10-26

First public release.

- **WRTDS model**: `model_estimation` with leave-one-out cross-validation,
  adaptive windows and edge adjustment as in current EGRET; interval-censored
  regression (`run_surv_reg`); daily estimates and flow normalization
  (`est_daily_from_surfaces`), including EGRET's pooling of February 28/29.
- **Generalized flow normalization** (`flex_fn`, `make_date_info`) and the
  wall (`stitch_surfaces`).
- **Trends**: `run_pairs`, `run_groups`, `run_series`, with flow breaks,
  walls, and sample and discharge date limits.
- **WRTDS-Kalman**: `wrtds_kalman`.
- **Annual and summary results**: `setup_years`, `calculate_monthly_results`,
  `table_results`, `table_change`, `error_stats`, `flux_bias_stat`,
  `blank_time`.
- **Data**: user files (`read_user_daily`, `read_user_sample`), USGS Water Data
  API readers, `merge_report`, `populate_daily`, censoring and QA helpers.
- **Streamflow trends**: annual flow statistics, flow duration, `cum_q_date`,
  Quantile-Kendall.
- **Plots**: 24 of EGRET's plots, each returning `(fig, ax)`.
- **Validation** against results computed by R EGRET 3.0.12 (see
  `docs/validation.md`).

Not yet included: EGRETci (bootstrap confidence intervals and p-values) and
some of EGRET's flow-history plots; see `docs/r_to_python.md`.

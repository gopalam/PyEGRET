# Changelog

## Unreleased

- **Tutorials**: Jupyter notebooks in `tutorials/`, readable on GitHub and
  runnable locally or in Google Colab. Tutorial 1 is a complete WRTDS analysis
  of the Choptank River; tutorial 2 covers preparing your own data. The
  Choptank data are included as plain CSV files.
- Tutorial 3 downloads a record from the USGS Water Data APIs and checks it
  against EGRET's.
- `diagnostics/run_tutorials.py` runs the notebooks; the test workflow now
  runs them on every push (the downloading ones in a separate job that is
  allowed to fail when USGS is unavailable), and `check_style.py` checks their
  code.
- **Fixed `read_waterdata_sample` against the live USGS Samples service**,
  which names its columns differently from what the reader assumed. It now
  follows R's `readNWISSample` step for step: the `fullphyschem` profile, R's
  renaming to Water Quality Portal names, `processQWData`'s non-detect
  wordings, `compressData` and `remove_zeros`. As in R it keeps every result,
  carries the descriptive columns (activity type, medium, start time and
  others), and reports characteristics that vary. A parameter given as a name
  rather than a five-digit code is queried as a characteristic.
- `read_waterdata_info` now takes the parameter's name and units from the USGS
  parameter-code table, as R does, warns when the units are not mg/L, and sets
  `station_nm`, `paStart` and `paLong`.
- `diagnostics/check_waterdata.py` compares a live download with EGRET's
  Choptank record; on 27 October 2026 the samples matched exactly and the
  discharge differed only by USGS's later rounding to three significant
  figures.

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

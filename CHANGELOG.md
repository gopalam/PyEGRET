# Changelog

## 0.2.0 - 2026-10-29

Adds the bootstrap uncertainty of EGRETci (`pyegret.ci`), fixes the USGS and
Water Quality Portal sample readers against the live services, and adds the
first three tutorials.

- **Uncertainty: EGRETci translated** (`pyegret.ci`). `run_pairs_boot` and
  `run_groups_boot` give bootstrap confidence intervals, two-sided p-values
  and likelihoods for `run_pairs` and `run_groups` results; `wbt` is the
  original WRTDS Bootstrap Test; `ci_calculations` gives confidence bands on
  the flow-normalized history, drawn by `plot_conc_hist_boot` and
  `plot_flux_hist_boot`; `plot_histogram_trend` shows the bootstrap
  distribution; `gen_daily_boot` with `make_daily_pi`, `make_month_pi` and
  `make_annual_pi` gives prediction intervals for WRTDS-Kalman estimates.
- Bootstrap resamples are drawn with R's own random-number generator, seeded
  as EGRETci seeds it, so each replicate matches EGRETci's one for one. Checked
  against EGRETci 2.0.5 run in R 4.3.3 (`tests/reference/egretci/`, with the R
  scripts): p-values and likelihoods identical, intervals within 0.3 to 0.6 %,
  confidence bands within 0.03 % (median).
- `gen_daily_boot` uses each bootstrap refit in its Kalman step, as EGRETci's
  documentation describes; EGRETci 2.0.5's `genDailyBoot` does not (its
  refits have no effect). `boot_surfaces=False` reproduces EGRETci.
- Fitting a surface now refuses a sample with no more observations than
  `min_num_obs` (or uncensored values than `min_num_uncen`), as R's
  `runSurvReg` does, instead of carrying on with wider windows.
- `model_estimation` records its settings in INFO (`windowY`, `windowQ`,
  `windowS`, `minNumObs`, `minNumUncen`, `edgeAdjust`), as R does, and
  `run_series` records `edgeAdjust` and marks its result with `runSeries`.
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
- **Fixed `read_wqp_sample`**, the Water Quality Portal reader for state,
  tribal and local agency data. It dropped every result reported as not
  detected (all 29 in one test record), which biases concentrations upward,
  and it ignored start and end dates. It now follows R's `readWQPSample`:
  non-detects become "less than" values at their detection limit, every
  result is kept with its descriptive columns, a message lists fractions,
  activity types and methods that vary, and USGS sites are passed to
  `read_waterdata_sample`. Two wordings R misses ("Below Detection Limit",
  "Below Reporting Limit") are also treated as non-detects, and a non-detect
  with no detection limit is named in a warning before it is dropped.
- The processing shared by the two sample readers is in
  `pyegret.data.qw_processing`; `process_wqp_results` mirrors R's
  `processQWData`.
- `read_nwis_sample` now calls `read_waterdata_sample`, as R's
  `readNWISSample` does, with a `FutureWarning`; it used to read the portal's
  copy of USGS data, which stopped updating in March 2024.
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

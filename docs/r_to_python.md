# R EGRET to PyEGRET: function map

Every R function PyEGRET translates, with its Python name and module. R's
camelCase names become snake_case (`modelEstimation` -> `model_estimation`),
arguments keep R's order and defaults (`windowSide` -> `window_side`), and
**data-frame column names are exactly R's** (`ConcDay`, `FNConc`, `paStart`).
A `-` in the R column marks a Python helper with no single R counterpart.

Most functions are also importable from their subpackage, e.g.
`from pyegret.analysis import run_pairs`, and the most used ones from the
top level: `from pyegret import merge_report, model_estimation, wrtds_kalman`.

## Reading and preparing data

| R | Python | module | what it does |
|---|---|---|---|
| `compressData` | `compress_data` | `pyegret.data.compress_data` | Combine remark/value column pairs into ConcLow, ConcHigh and Uncen. |
| `removeDuplicates` | `remove_duplicates` | `pyegret.data.compress_data` | Drop samples that repeat an earlier sample's date and ConcHigh. |
| `remove_zeros (internal)` | `remove_zeros` | `pyegret.data.compress_data` | Drop rows whose concentration was reported as exactly 0. |
| `mergeReport` | `merge_report` | `pyegret.data.merge_report` | Validate and join site metadata, daily discharge, and WQ samples into an eList. |
| `mergeReport (merge step)` | `merge_daily_q` | `pyegret.data.populate` | Merge daily discharge (Q, LogQ) onto the Sample DataFrame by date. |
| `populateDaily` | `populate_daily` | `pyegret.data.populate` | Build an EGRET Daily DataFrame from raw dates and discharge values. |
| `populateDateColumns` | `populate_date_columns` | `pyegret.data.populate` | Add standard EGRET date columns derived from the ``Date`` column. |
| `populateSampleColumns` | `populate_sample_columns` | `pyegret.data.populate` | Add derived concentration columns to a Sample DataFrame. |
| `dataOverview` | `data_overview` | `pyegret.data.quality` | Summarise a record before modelling it, and flag what looks wrong. |
| `fixSampleFrame` | `fix_sample_frame` | `pyegret.data.quality` | Recompute ``ConcAve`` and ``Uncen`` from the censoring bounds. |
| `makeAugmentedSample` | `make_augmented_sample` | `pyegret.data.quality` | Draw a plausible value for each censored sample, for plotting. |
| `populateConcentrations` | `populate_concentrations` | `pyegret.data.quality` | Turn reported values and remark codes into EGRET's censoring bounds. |
| `processQWData` | `process_wqp_results` | `pyegret.data.qw_processing` | Mark non-detects and choose each result's value. |
| - | `wqp_results_to_sample` | `pyegret.data.qw_processing` | Turn Water Quality Portal results into an EGRET Sample (the steps readNWISSample and readWQPSample share). |
| - | `process_qw_data` | `pyegret.data.quality` | Build an EGRET Sample frame from your own table of dates, values and remarks. |
| `readNWISDaily` | `read_nwis_daily` | `pyegret.data.read_nwis` | Legacy NWIS services; use `read_waterdata_daily`. Fetch USGS NWIS mean-daily values and return an EGRET Daily DataFrame. |
| `readNWISInfo` | `read_nwis_info` | `pyegret.data.read_nwis` | Legacy NWIS services; use `read_waterdata_info`. Fetch site and parameter metadata from USGS NWIS. |
| `readNWISSample` | `read_nwis_sample` | `pyegret.data.read_nwis` | Kept for older code: now calls `read_waterdata_sample`, with a warning. |
| `readDataFromFile` | `read_data_from_file` | `pyegret.data.read_user` | Read a delimited file whose first column is a date. |
| `readUserDaily` | `read_user_daily` | `pyegret.data.read_user` | Read a two-column daily discharge file into an EGRET Daily DataFrame. |
| `readUserSample` | `read_user_sample` | `pyegret.data.read_user` | Read a water-quality file into an EGRET Sample DataFrame. |
| `readNWISDaily` | `read_waterdata_daily` | `pyegret.data.read_waterdata` | Fetch mean daily values and return an EGRET Daily DataFrame. |
| `readNWISInfo` | `read_waterdata_info` | `pyegret.data.read_waterdata` | Fetch monitoring-location metadata as a one-row INFO frame. |
| `readNWISSample` | `read_waterdata_sample` | `pyegret.data.read_waterdata` | Fetch discrete water-quality samples and return an EGRET Sample frame. |
| - | `usgs_location_id` | `pyegret.data.read_waterdata` | Return a monitoring location id in the modern ``USGS-#####`` form. |
| `readWQPSample` | `read_wqp_sample` | `pyegret.data.read_wqp` | Fetch water-quality results from the Water Quality Portal as an EGRET Sample. |

## The eList

| R | Python | module | what it does |
|---|---|---|---|
| `blankTime` | `blank_time` | `pyegret.elist` | Blank out the model results over a stretch with no samples. |
| - | `describe_elist` | `pyegret.elist` | Return a short text summary of an eList (site, dates, samples, model). |
| - | `has_model` | `pyegret.elist` | Return True if WRTDS surfaces have been estimated. |
| `is.egret` | `has_sample` | `pyegret.elist` | Return True if the eList holds a non-empty Sample DataFrame. |
| `as.egret` | `make_elist` | `pyegret.elist` | Build an eList dict from its parts. |
| `nCensoredVals` | `n_censored_vals` | `pyegret.elist` | Number of censored samples. Mirrors R ``nCensoredVals()``. |
| `nDischarge` | `n_discharge` | `pyegret.elist` | Number of daily discharge values. Mirrors R ``nDischarge()``. |
| `nObservations` | `n_observations` | `pyegret.elist` | Number of samples. Mirrors R ``nObservations()``; None without a Sample. |

## The WRTDS model

| R | Python | module | what it does |
|---|---|---|---|
| `estCrossVal` | `est_cross_val` | `pyegret.model.est_cross_val` | Leave-one-out cross-validated WRTDS predictions for every sample. |
| `estDailyFromSurfaces` | `est_daily_from_surfaces` | `pyegret.model.est_daily` | Interpolate the WRTDS surface onto every row of Daily. |
| `surfaceIndex` | `build_grid_axes` | `pyegret.model.est_surfaces` | Build the (year, logQ) axes the surface is estimated on. |
| `decimalHighLow` | `decimal_high_low` | `pyegret.model.est_surfaces` | Water-year bounds of a sample record, for edge adjustment. |
| `estSurfaces` | `est_surfaces` | `pyegret.model.est_surfaces` | Estimate the WRTDS concentration surface over a (year, logQ) grid. |
| `estSurfaces` | `est_surfaces_on_grid` | `pyegret.model.est_surfaces` | Fit the WRTDS regression at every point of a given (year, logQ) grid. |
| `runSurvReg` | `run_surv_reg_points` | `pyegret.model.est_surfaces` | Fit the WRTDS regression at any list of (year, logQ) points. |
| `flexFN` | `flex_fn` | `pyegret.model.flex_fn` | Flow-normalize each stretch of the record against its own flows. |
| `runSeries` | `make_date_info` | `pyegret.model.flex_fn` | Build the segment table that says how each stretch is normalized. |
| `cleanUp` | `clean_up` | `pyegret.model.kalman` | One sample per day, with censored values replaced by random draws. |
| `genmissing` | `gen_missing` | `pyegret.model.kalman` | Simulate an AR(1) segment between two known end values. |
| `populateDailySamp` | `populate_daily_samp` | `pyegret.model.kalman` | Daily with the sampled concentration and standardized residual added. |
| `precompute_L` | `precompute_l` | `pyegret.model.kalman` | Cholesky factor of the precision matrix of an AR(1) bridge. |
| `randomSubset` | `random_subset` | `pyegret.model.kalman` | Keep one randomly chosen row for each repeated value of a column. |
| `WRTDSKalman` | `wrtds_kalman` | `pyegret.model.kalman` | Add WRTDS-Kalman estimates ``GenConc`` and ``GenFlux`` to Daily. |
| `modelEstimation` | `model_estimation` | `pyegret.model.model_estimation` | Run the complete WRTDS model estimation pipeline. |
| `runSurvReg` | `run_surv_reg` | `pyegret.model.run_surv_reg` | Fit a weighted censored log-normal regression at one WRTDS grid point. |
| `runSurvReg` | `compute_adaptive_weights` | `pyegret.model.tri_cube` | Tricube weights with EGRET's window expansion and edge adjustment. |
| - | `compute_weights` | `pyegret.model.tri_cube` | Combined WRTDS tricube weights for all sample observations. |
| - | `season_distance` | `pyegret.model.tri_cube` | Seasonal distance in years, in ``[0, 0.5]``. |
| `triCube` | `tri_cube` | `pyegret.model.tri_cube` | Tricube kernel weight function. |
| `stitch` | `stitch_surfaces` | `pyegret.model.wall` | Fit the WRTDS surface separately either side of a date, then join them. |

## Annual results, tables and trends

| R | Python | module | what it does |
|---|---|---|---|
| `errorStats` | `error_stats` | `pyegret.analysis.error_stats` | Goodness-of-fit statistics for a fitted WRTDS model. |
| `fluxBiasStat` | `flux_bias_stat` | `pyegret.analysis.error_stats` | The flux bias statistic. Mirrors R ``fluxBiasStat()``. |
| `calculateMonthlyResults` | `calculate_monthly_results` | `pyegret.analysis.monthly_results` | Average the daily results within each calendar month of the record. |
| `printGroups` | `print_groups` | `pyegret.analysis.run_groups` | Print a run_groups result as R does. Mirrors R ``printGroups()``. |
| `runGroups` | `run_groups` | `pyegret.analysis.run_groups` | Change between the averages of two groups of years. Mirrors R ``runGroups()``. |
| - | `span_surface_elist` | `pyegret.analysis.run_groups` | The set-up R's runGroups and runSeries share. |
| `printPairs` | `print_pairs` | `pyegret.analysis.run_pairs` | Print a run_pairs result as R does. Mirrors R ``printPairs()``. |
| `runPairs` | `run_pairs` | `pyegret.analysis.run_pairs` | Change in flow-normalized concentration and flux between two years. |
| `runSeries` | `run_series` | `pyegret.analysis.run_series` | Flow-normalized daily and annual series with a moving flow window. |
| - | `get_pa` | `pyegret.analysis.setup_years` | The period of analysis to use: explicit values, else INFO, else 10/12. |
| `setPA` | `set_pa` | `pyegret.analysis.setup_years` | Record the period of analysis in INFO. Mirrors R ``setPA()``. |
| `setupYears` | `setup_years` | `pyegret.analysis.setup_years` | Annual means of the daily WRTDS results. Mirrors R ``setupYears()``. |
| `startEnd` | `start_end` | `pyegret.analysis.setup_years` | First and last date of one period of analysis. Mirrors R ``startEnd()``. |
| `tableChange` | `table_change` | `pyegret.analysis.table_change` | Concentration and flux change tables. Mirrors R ``tableChange()``. |
| `tableChangeSingle` | `table_change_single` | `pyegret.analysis.table_change` | Change in flow-normalized concentration or flux between pairs of years. |
| `tableResults` | `table_results` | `pyegret.analysis.table_change` | Table of annual discharge, concentration and flux. |
| - | `add_plus` | `pyegret.analysis.trend_setup` | Signed percent with two decimals, as R's internal ``add_plus()``. |
| `decimalDate` | `decimal_date` | `pyegret.analysis.trend_setup` | Decimal year of one date. Mirrors R ``decimalDate()`` for a scalar. |
| `estSurfaces` | `estimate_span_surface` | `pyegret.analysis.trend_setup` | Fit the surface only for the years between two dates. |
| - | `flow_date_info` | `pyegret.analysis.trend_setup` | The flow-normalization segments ``runGroups`` and ``runSeries`` build. |
| - | `record_dates` | `pyegret.analysis.trend_setup` | First and last sample and discharge dates, defaulting to the record's ends. |
| `surfaceStartEnd` | `surface_start_end` | `pyegret.analysis.trend_setup` | Whole periods of analysis covering two dates. Mirrors R ``surfaceStartEnd()``. |

## Plots

| R | Python | module | what it does |
|---|---|---|---|
| `boxConcMonth` | `box_conc_month` | `pyegret.plots.data_plots` | Box plots of sampled concentration (ConcAve) by month. Mirrors R ``boxConcMonth()``. |
| `boxQTwice` | `box_q_twice` | `pyegret.plots.data_plots` | Discharge on sampled days beside discharge on all days. Mirrors R ``boxQTwice()``. |
| `multiPlotDataOverview` | `multi_plot_data_overview` | `pyegret.plots.data_plots` | Four views of the raw data on one page. Mirrors R ``multiPlotDataOverview()``. |
| `plotConcQ` | `plot_conc_q` | `pyegret.plots.data_plots` | Concentration against discharge, discharge on a log axis. Mirrors R ``plotConcQ()``. |
| `plotConcTime` | `plot_conc_time` | `pyegret.plots.data_plots` | Concentration against time, optionally for a range of discharge. |
| `plotQTimeDaily` | `plot_q_time_daily` | `pyegret.plots.data_plots` | Daily discharge over time. Mirrors R ``plotQTimeDaily()``. |
| `boxConcThree` | `box_conc_three` | `pyegret.plots.diagnostic_plots` | Box plots of sampled values, estimates on sampled days, and all days. |
| `boxResidMonth` | `box_resid_month` | `pyegret.plots.diagnostic_plots` | Box plots of residuals by month. Mirrors R ``boxResidMonth()``. |
| `fluxBiasMulti` | `flux_bias_multi` | `pyegret.plots.diagnostic_plots` | Eight diagnostic panels and the flux bias statistic. Mirrors R ``fluxBiasMulti()``. |
| `plotConcPred` | `plot_conc_pred` | `pyegret.plots.diagnostic_plots` | Observed against estimated concentration, with a 1:1 line. |
| `plotFluxPred` | `plot_flux_pred` | `pyegret.plots.diagnostic_plots` | Observed against estimated flux on the sampled days. Mirrors R ``plotFluxPred()``. |
| `plotResidPred` | `plot_resid_pred` | `pyegret.plots.diagnostic_plots` | Residual against estimated log concentration. Mirrors R ``plotResidPred()``. |
| `plotResidQ` | `plot_resid_q` | `pyegret.plots.diagnostic_plots` | Residual against discharge (log axis). Mirrors R ``plotResidQ()``. |
| `plotResidTime` | `plot_resid_time` | `pyegret.plots.diagnostic_plots` | Residual against time. Mirrors R ``plotResidTime()``. |
| `plotConcHist` | `plot_conc_hist` | `pyegret.plots.history_plots` | Annual mean and flow-normalized concentration. Mirrors R ``plotConcHist()``. |
| `plotConcTimeDaily` | `plot_conc_time_daily` | `pyegret.plots.history_plots` | Daily estimated concentration with the observations. Mirrors R ``plotConcTimeDaily()``. |
| `plotFluxHist` | `plot_flux_hist` | `pyegret.plots.history_plots` | Annual mean and flow-normalized flux. Mirrors R ``plotFluxHist()``. |
| `plotFluxTimeDaily` | `plot_flux_time_daily` | `pyegret.plots.history_plots` | Daily estimated flux with the observed flux. Mirrors R ``plotFluxTimeDaily()``. |
| `plotTimeSlice` | `plot_time_slice` | `pyegret.plots.history_plots` | WRTDS and WRTDS-Kalman daily estimates over a short stretch, with the data. |
| `plotWRTDSKalman` | `plot_wrtds_kalman` | `pyegret.plots.history_plots` | Annual WRTDS and WRTDS-Kalman flux, over time and against each other. |
| `plotConcQSmooth` | `plot_conc_q_smooth` | `pyegret.plots.surface_plots` | Modelled concentration against discharge on up to three dates. |
| `plotConcTimeSmooth` | `plot_conc_time_smooth` | `pyegret.plots.surface_plots` | Modelled concentration over time at up to three discharges. |
| `plotContours` | `plot_contours` | `pyegret.plots.surface_plots` | Filled contours of the WRTDS surface over time and discharge. |
| `plotDiffContours` | `plot_diff_contours` | `pyegret.plots.surface_plots` | Change in the surface from one year to another, by season and discharge. |

## Streamflow trends (flowHistory)

| R | Python | module | what it does |
|---|---|---|---|
| - | `add_rolling_means` | `pyegret.flow.annual_series` | Add the trailing 7-day and 30-day mean discharge columns. |
| `cumQdate` | `cum_q_date` | `pyegret.flow.annual_series` | The day each year by which a given fraction of runoff has passed. |
| `flowDuration` | `flow_duration` | `pyegret.flow.annual_series` | Flow duration curve, optionally restricted to a season. |
| `makeAnnualSeries` | `make_annual_series` | `pyegret.flow.annual_series` | Compute and smooth the eight annual flow statistics. |
| - | `plot_flow_trend` | `pyegret.flow.plots` | Plot one annual flow statistic with its smoothed trend. |
| - | `plot_quantile_kendall` | `pyegret.flow.plots` | Plot the trend in every order statistic against its plotting position. |
| - | `quantile_kendall` | `pyegret.flow.quantile_kendall` | Trend in every order statistic of the annual flow distribution. |
| - | `sort_annual_flows` | `pyegret.flow.quantile_kendall` | Stack each complete year's sorted daily discharges into a matrix. |
| - | `mann_kendall` | `pyegret.flow.trend_tests` | Mann-Kendall trend test with tie and continuity corrections. |
| - | `theil_sen_slope` | `pyegret.flow.trend_tests` | Median of the slopes between every pair of points. |
| - | `zhang_prewhitened_test` | `pyegret.flow.trend_tests` | Mann-Kendall test after Zhang's iterative prewhitening. |

## Units, dates and labels

| R | Python | module | what it does |
|---|---|---|---|
| `dateFormatCheck` | `check_date_format` | `pyegret.utils.date_utils` | Return True if date_str is a valid YYYY-MM-DD string or empty string. |
| - | `day_of_water_year` | `pyegret.utils.date_utils` | Return EGRET's leap-adjusted day-of-year, in which March 1 is always 61. |
| - | `julian_day` | `pyegret.utils.date_utils` | Return integer days elapsed since 1850-01-01 (EGRET convention). |
| - | `month_seq` | `pyegret.utils.date_utils` | Return EGRET's absolute month sequence, ``(year - 1850) * 12 + month``. |
| `decimalDate` | `to_decimal_year` | `pyegret.utils.date_utils` | Convert a Series of dates to decimal year. |
| - | `water_year` | `pyegret.utils.date_utils` | Return the USGS water year: October through September, labelled by its end. |
| `setSeasonLabel` | `set_season_label` | `pyegret.utils.season` | Return EGRET's label for a period of analysis. |
| `setSeasonLabelByUser` | `set_season_label_by_user` | `pyegret.utils.season` | Same as :func:`set_season_label`. Mirrors R ``setSeasonLabelByUser()``. |
| `fluxConst` | `get_flux_unit` | `pyegret.utils.units` | Return the flux-unit entry for an EGRET fluxUnit number or name. |
| `qConst` | `get_q_unit` | `pyegret.utils.units` | Return the discharge-unit entry for an EGRET qUnit number or name. |
| `printFluxUnitCheatSheet` | `print_flux_unit_cheat_sheet` | `pyegret.utils.units` | Print the flux units. Mirrors R ``printFluxUnitCheatSheet()``. |
| `printqUnitCheatSheet` | `print_q_unit_cheat_sheet` | `pyegret.utils.units` | Print the discharge units. Mirrors R ``printqUnitCheatSheet()``. |

## Not yet translated

**EGRETci** (bootstrap confidence intervals and trend p-values): `wBT`,
`runPairsBoot`, `runGroupsBoot`, `ciCalculations`, `ciBands`,
`plotConcHistBoot`, `plotFluxHistBoot`, `genDailyBoot` and the prediction
intervals. This is the next major piece of work.

**EGRET**: `plotFlowSingle`, `plotFour`, `plotFourStats`, `plotSDLogQ`,
`tableFlowChange`, `printSeries` (PyEGRET's `plot_flow_trend` and the
Quantile-Kendall figure cover similar ground), `plotFluxQ`, `plotMonthTrend`,
`plot15`, `plot1of15`, `fill_missing_daily`, `readUserInfo`, `readWQPInfo`,
`populateSiteINFO`, `populateParameterINFO`, `saveResults`, `run_WRTDS`,
`checkSurfaceSpan`, `jitterSam`. R's internal plotting helpers
(`generalAxis`, `genericEGRETDotPlot`, `yPretty`, `flexPlotAddOn`) are not
needed; matplotlib does that work.

The accessor functions `getDaily`, `getSample`, `getInfo` and `getSurfaces`
are not needed either: the eList is a dict, so use `elist["Daily"]` and so on.

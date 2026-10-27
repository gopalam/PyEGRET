"""
Leave-one-out cross-validation of the WRTDS model.

Mirrors R EGRET's ``estCrossVal()``.

The surface is fit from every sample, so predicting a sample from that surface
asks the model to reproduce a value it was told. Residuals built that way
flatter the model: the error statistics come out too small and the flux bias
statistic too close to zero.

Cross-validation removes one observation, refits the weighted regression at
that observation's own (DecYear, LogQ), and predicts it from the remaining
data. Repeat for every sample and the predictions are honest - they never saw
the value they are predicting. These are what EGRET stores in
``Sample$yHat``, ``SE`` and ``ConcHat``, and what ``errorStats`` and
``fluxBiasStat`` are defined on.

Note the edge-adjustment reference dates differ from ``est_surfaces``:
``modelEstimation()`` passes the **Daily** record bounds here, while
``estSurfaces()`` uses ``decimalHighLow()`` of the Sample. This module
follows EGRET on both counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.elist import has_sample
from pyegret.model.run_surv_reg import run_surv_reg
from pyegret.model.tri_cube import (
    DEFAULT_MIN_NUM_OBS,
    DEFAULT_MIN_NUM_UNCEN,
    DEFAULT_WINDOW_Q,
    DEFAULT_WINDOW_S,
    DEFAULT_WINDOW_Y,
    compute_adaptive_weights,
)

__all__ = ["est_cross_val"]

_TWO_PI = 2.0 * np.pi


def est_cross_val(
    elist: dict,
    window_y: float = DEFAULT_WINDOW_Y,
    window_q: float = DEFAULT_WINDOW_Q,
    window_s: float = DEFAULT_WINDOW_S,
    min_num_obs: int = DEFAULT_MIN_NUM_OBS,
    min_num_uncen: int = DEFAULT_MIN_NUM_UNCEN,
    edge_adjust: bool = True,
    dec_low: float | None = None,
    dec_high: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-one-out cross-validated WRTDS predictions for every sample.

    Mirrors R ``estCrossVal()``. Fits the model ``n`` times, once per
    observation, each time excluding the observation being predicted.

    Args:
        elist: eList dict with Daily and Sample DataFrames.
        window_y: Starting tricube half-window for time, in years. Default 7.0.
        window_q: Starting half-window for log-discharge. Default 2.0.
        window_s: Starting half-window for season, in years. Default 0.5.
        min_num_obs: Minimum weighted observations per fit. Default 100.
        min_num_uncen: Minimum uncensored weighted observations. Default 50.
        edge_adjust: Widen the time window near the record ends. Default True.
        dec_low: Start of the record for edge adjustment. Defaults to the
            first Daily decimal year, as ``modelEstimation()`` passes.
        dec_high: End of the record. Defaults to the last Daily decimal year.
        verbose: Print progress. Default True.

    Returns:
        A copy of Sample with ``yHat``, ``SE``, ``ConcHat`` and ``resid``
        replaced by cross-validated values. ``resid`` is
        ``log(ConcAve) - yHat``, the residual R EGRET's residual plots use
        when not randomising censored values (``boxResidMonth`` and friends
        with ``randomCensored = FALSE``).

    Raises:
        ValueError: If ``elist`` has no Sample data.
    """
    if not has_sample(elist):
        raise ValueError("eList has no Sample data; cannot cross-validate.")

    sample = elist["Sample"].reset_index(drop=True)
    n = len(sample)

    if dec_low is None or dec_high is None:
        daily_years = elist["Daily"]["DecYear"].to_numpy(dtype=float)
        dec_low = float(daily_years[0]) if dec_low is None else dec_low
        dec_high = float(daily_years[-1]) if dec_high is None else dec_high

    years = sample["DecYear"].to_numpy(dtype=float)
    log_q = sample["LogQ"].to_numpy(dtype=float)
    uncen = sample["Uncen"].to_numpy()

    y_hat = np.full(n, np.nan)
    se = np.full(n, np.nan)
    conc_hat = np.full(n, np.nan)

    if verbose:
        print(f"Cross-validating {n} samples...", flush=True)
    next_report = 10

    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False

        weights_kept, _ = compute_adaptive_weights(
            target_year=years[i],
            target_logQ=log_q[i],
            sample_years=years[keep],
            sample_logQ=log_q[keep],
            sample_uncen=uncen[keep],
            window_y=window_y,
            window_q=window_q,
            window_s=window_s,
            min_num_obs=min_num_obs,
            min_num_uncen=min_num_uncen,
            edge_adjust=edge_adjust,
            dec_low=dec_low,
            dec_high=dec_high,
        )

        # Expand back to full length with a hard zero on the held-out row, so
        # run_surv_reg (which drops zero-weight rows) never sees it.
        weights = np.zeros(n)
        weights[keep] = weights_kept

        y_hat[i], se[i], conc_hat[i] = run_surv_reg(
            years[i], log_q[i],
            float(np.sin(_TWO_PI * years[i])),
            float(np.cos(_TWO_PI * years[i])),
            sample, weights,
        )

        if verbose:
            pct = int(100 * (i + 1) / n)
            if pct >= next_report:
                print(f"  {next_report}% complete", flush=True)
                next_report += 10

    sample["yHat"] = y_hat
    sample["SE"] = se
    sample["ConcHat"] = conc_hat
    with np.errstate(divide="ignore"):
        log_conc_ave = np.log(sample["ConcAve"].to_numpy(dtype=float))
    sample["resid"] = log_conc_ave - y_hat

    n_failed = int(np.sum(np.isnan(y_hat)))
    if verbose and n_failed:
        print(f"  {n_failed} of {n} samples could not be cross-validated.",
              flush=True)

    return sample

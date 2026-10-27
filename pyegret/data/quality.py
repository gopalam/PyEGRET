"""
Data quality utilities for water-quality records.

Everything here is about getting a Sample frame into honest shape before any
model sees it, and about looking at what you actually have. None of it depends
on WRTDS.

The centre of gravity is censored data. A laboratory result of "< 0.05" is not
a measurement of 0.05, and it is not a missing value either: it is the
statement that the true concentration lies somewhere in [0, 0.05]. EGRET
carries that as a pair of bounds - ``ConcLow`` and ``ConcHigh`` - with
``Uncen`` marking whether they coincide. These functions build that
representation from the remark codes laboratories actually report, repair it
when it has been damaged, and - through :func:`make_augmented_sample` - give
you a defensible way to *plot* censored values without pretending they are
points.

Mirrors R EGRET's ``populateConcentrations``, ``processQWData``,
``fixSampleFrame``, ``dataOverview`` and ``makeAugmentedSample``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyegret.elist import has_sample
from pyegret.utils.convert import as_dates, as_float_array, clean_text

__all__ = [
    "populate_concentrations",
    "process_qw_data",
    "fix_sample_frame",
    "data_overview",
    "make_augmented_sample",
]

# Remark codes as reported by USGS and most laboratories.
LESS_THAN = "<"
GREATER_THAN = ">"


def populate_concentrations(
    values,
    remark_codes=None,
) -> pd.DataFrame:
    """Turn reported values and remark codes into EGRET's censoring bounds.

    Mirrors R ``populateConcentrations()``.

    ==============  =========  ==========  =========  =======
    remark          meaning    ConcLow     ConcHigh   Uncen
    ==============  =========  ==========  =========  =======
    none or blank   detected   value       value      1
    ``"<"``         below      0           value      0
    ``">"``         above      value       0          0
    ==============  =========  ==========  =========  =======

    The ``">"`` convention looks odd out of context: EGRET stores a
    right-censored value by zeroing ``ConcHigh``, which flags the row as
    censored rather than describing an upper bound. WRTDS is built around
    left-censoring, so right-censored results are rare and are carried, not
    modelled.

    Args:
        values: Reported concentrations. Missing values become 0, as in EGRET.
        remark_codes: Matching remark codes. None means every value is
            a detection.

    Returns:
        DataFrame with ``ConcLow``, ``ConcHigh`` and ``Uncen`` columns.
    """
    values = pd.Series(as_float_array(values))
    values = values.fillna(0.0)

    if remark_codes is None:
        codes = pd.Series([""] * len(values))
    else:
        codes = clean_text(remark_codes)

    below = codes == LESS_THAN
    above = codes == GREATER_THAN

    conc_low = values.where(~below, 0.0)
    conc_high = values.where(~above, 0.0)
    # As in R: a zero at either end marks the row censored. That includes a
    # reported value of 0.0, which remove_zeros() later drops.
    uncen = np.where((conc_low == 0) | (conc_high == 0), 0, 1)

    return pd.DataFrame({
        "ConcLow": conc_low.to_numpy(dtype=float),
        "ConcHigh": conc_high.to_numpy(dtype=float),
        "Uncen": np.asarray(uncen, dtype=int),
    })


def process_qw_data(
    frame: pd.DataFrame,
    date_col: str = "Date",
    value_col: str = "value",
    remark_col: str | None = "remark_cd",
) -> pd.DataFrame:
    """Build an EGRET Sample frame from a table of dates, values and remarks.

    Mirrors R ``processQWData()``. This is the entry point for laboratory
    exports and monitoring databases that report a concentration alongside a
    remark code, rather than pre-split bounds.

    Args:
        frame: Source table.
        date_col: Column holding sample dates.
        value_col: Column holding reported concentrations.
        remark_col: Column holding remark codes, or None if the record has
            no censoring.

    Returns:
        DataFrame with ``Date``, ``ConcLow``, ``ConcHigh`` and ``Uncen``,
        sorted by date, with undated rows dropped. Pass it through
        :func:`~pyegret.data.populate.populate_date_columns` and
        :func:`~pyegret.data.populate.populate_sample_columns` to complete it.

    Raises:
        KeyError: If a named column is not present.
    """
    for name in (date_col, value_col):
        if name not in frame.columns:
            raise KeyError(
                f"Column {name!r} not in the frame. Available: {list(frame.columns)}"
            )
    if remark_col is not None and remark_col not in frame.columns:
        raise KeyError(
            f"Remark column {remark_col!r} not in the frame. "
            f"Available: {list(frame.columns)}. Pass remark_col=None if the "
            "record has no censoring."
        )

    codes = frame[remark_col] if remark_col is not None else None
    bounds = populate_concentrations(frame[value_col], codes)

    dates = as_dates(frame[date_col])
    sample = pd.DataFrame({
        "Date": dates.to_numpy(),
        "ConcLow": bounds["ConcLow"].to_numpy(),
        "ConcHigh": bounds["ConcHigh"].to_numpy(),
        "Uncen": bounds["Uncen"].to_numpy(),
    })
    sample = sample.dropna(subset=["Date"])
    sample = sample.sort_values("Date")
    sample = sample.reset_index(drop=True)
    return sample


def fix_sample_frame(elist) -> dict:  # noqa: F821
    """Recompute ``ConcAve`` and ``Uncen`` from the censoring bounds.

    Mirrors R ``fixSampleFrame()``. Use it after editing a Sample frame by
    hand, or after merging records from sources that disagree about how to
    spell a non-detect: it derives the two dependent columns from
    ``ConcLow`` and ``ConcHigh`` alone, so they cannot drift out of step with
    the bounds.

    A missing or zero ``ConcLow`` means no known lower bound, so ``ConcAve``
    becomes half of ``ConcHigh``. ``Uncen`` is 1 only where the two bounds
    coincide.

    Args:
        elist: eList dict with a Sample DataFrame.

    Returns:
        A new eList with Sample repaired; the input is not changed.

    Raises:
        ValueError: If the eList has no Sample data.
    """
    if not has_sample(elist):
        raise ValueError("eList has no Sample data.")

    elist = dict(elist)
    sample = elist["Sample"].copy()
    conc_low = as_float_array(sample["ConcLow"])
    conc_high = as_float_array(sample["ConcHigh"])

    no_lower_bound = ~np.isfinite(conc_low) | (conc_low == 0.0)
    conc_ave = np.where(no_lower_bound, conc_high / 2.0, (conc_low + conc_high) / 2.0)

    sample["ConcAve"] = conc_ave
    sample["Uncen"] = (conc_high == conc_ave).astype(int)

    elist["Sample"] = sample
    return elist


def data_overview(elist=None, daily=None, sample=None, verbose: bool = True) -> dict:
    """Summarise a record before modelling it, and flag what looks wrong.

    Mirrors R ``dataOverview()``, but returns the summary as well as printing
    it, so it can be run across a monitoring network and the results collected
    into a table.

    Checks the two ordering problems that break WRTDS silently: samples
    before the discharge record starts, and samples after it ends. Either
    means the model would be asked to predict from discharge it does not have.

    Args:
        elist: eList dict with Daily and Sample. Alternatively pass ``daily`` and
            ``sample`` directly.
        daily: Daily DataFrame, if not passing an eList.
        sample: Sample DataFrame, if not passing an eList.
        verbose: Print the summary. Default True.

    Returns:
        Dictionary with record extents, discharge and concentration ranges,
        the censored fraction, and a ``problems`` list naming any issue
        found. An empty ``problems`` list means the record is usable.

    Raises:
        ValueError: If neither an eList nor both frames are given.
    """
    if elist is not None:
        daily, sample = elist["Daily"], elist["Sample"]
    if daily is None or sample is None:
        raise ValueError("Pass an eList, or both daily and sample.")

    daily = daily.sort_values("Date")
    sample = sample.sort_values("Date")

    n_days, n_samples = len(daily), len(sample)
    q = pd.to_numeric(daily["Q"], errors="coerce")
    conc = pd.to_numeric(sample["ConcHigh"], errors="coerce")
    n_censored = int((sample["Uncen"] == 0).sum())

    summary = {
        "n_days": n_days,
        "n_years": round(n_days / 365.25, 1),
        "daily_start": daily["Date"].iloc[0],
        "daily_end": daily["Date"].iloc[-1],
        "n_samples": n_samples,
        "sample_start": sample["Date"].iloc[0] if n_samples else None,
        "sample_end": sample["Date"].iloc[-1] if n_samples else None,
        "samples_per_year": round(n_samples / max(n_days / 365.25, 1e-9), 1),
        "q_min": float(q.min()), "q_mean": float(q.mean()), "q_max": float(q.max()),
        "conc_min": float(conc.min()), "conc_mean": float(conc.mean()),
        "conc_max": float(conc.max()),
        "n_censored": n_censored,
        "pct_censored": round(100.0 * n_censored / n_samples, 1) if n_samples else 0.0,
        "problems": [],
    }

    if n_samples:
        if summary["sample_start"] < summary["daily_start"]:
            summary["problems"].append(
                "first sample predates the discharge record")
        if summary["sample_end"] > summary["daily_end"]:
            summary["problems"].append(
                "last sample postdates the discharge record")
    if (q <= 0).any():
        summary["problems"].append(
            f"{int((q <= 0).sum())} days have non-positive discharge")
    if summary["pct_censored"] > 50:
        summary["problems"].append(
            f"{summary['pct_censored']}% of samples are censored")

    if verbose:
        print(f"Discharge record: {n_days} days ({summary['n_years']} years), "
              f"{summary['daily_start'].date()} to {summary['daily_end'].date()}")
        print(f"  Q (min / mean / max): {summary['q_min']:.4g} / "
              f"{summary['q_mean']:.4g} / {summary['q_max']:.4g}")
        print(f"Water-quality record: {n_samples} samples "
              f"({summary['samples_per_year']} per year)")
        if n_samples:
            print(f"  {summary['sample_start'].date()} to "
                  f"{summary['sample_end'].date()}")
            print(f"  concentration (min / mean / max): {summary['conc_min']:.3g} / "
                  f"{summary['conc_mean']:.3g} / {summary['conc_max']:.3g}")
            print(f"  censored: {n_censored} of {n_samples} "
                  f"({summary['pct_censored']}%)")
        for problem in summary["problems"]:
            print(f"  PROBLEM: {problem}")

    return summary


def make_augmented_sample(elist, seed: int | None = None) -> dict:  # noqa: F821
    """Draw a plausible value for each censored sample, for plotting.

    Mirrors R ``makeAugmentedSample()``. Requires ``yHat`` and ``SE`` on
    Sample, so run ``model_estimation`` first.

    A censored result cannot honestly be plotted as a point: putting it at the
    reporting limit overstates it, at zero understates it, and at half the
    limit invents a precision nobody has. This draws a random value from the
    fitted model's own distribution, *truncated to the interval the laboratory
    actually reported*. Across a whole record the cloud of drawn values has
    the right shape, so residual and concentration plots are honest about
    where censored data sits without claiming to know each value.

    The draws are random. Two runs give different pictures, which is the
    point: anything that changes between runs was never in the data. Pass
    ``seed`` for a reproducible figure.

    Args:
        elist: dict whose Sample carries ``yHat`` and ``SE``.
        seed: Seed for reproducible draws. None draws fresh.

    Returns:
        A new eList (the input is not changed), with ``rResid`` (drawn
        log-scale residual) and
        ``rObserved`` (the implied concentration) added to Sample.

    Raises:
        ValueError: If the model has not been run.
    """
    from scipy.stats import truncnorm

    if not has_sample(elist):
        raise ValueError("eList has no Sample data.")
    elist = dict(elist)
    sample = elist["Sample"].copy()
    missing = {"yHat", "SE"} - set(sample.columns)
    if missing:
        raise ValueError(
            f"Sample is missing {sorted(missing)}; run model_estimation() first."
        )

    y_hat = sample["yHat"].to_numpy(dtype=float)
    se = sample["SE"].to_numpy(dtype=float)
    uncen = sample["Uncen"].to_numpy()
    conc_low = as_float_array(sample["ConcLow"])
    conc_high = as_float_array(sample["ConcHigh"])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_high = np.log(conc_high)
        log_low = np.where(conc_low > 0, np.log(conc_low), -np.inf)

    # Detected values have no interval to draw from: the residual is exact.
    r_resid = log_high - y_hat

    censored = (uncen == 0) & np.isfinite(se) & (se > 0) & np.isfinite(y_hat)
    if censored.any():
        lower = log_low[censored] - y_hat[censored]
        upper = log_high[censored] - y_hat[censored]
        scale = se[censored]
        a = np.where(np.isfinite(lower), lower / scale, -np.inf)
        b = upper / scale
        r_resid[censored] = truncnorm.rvs(
            a, b, loc=0.0, scale=scale,
            size=int(censored.sum()),
            random_state=np.random.default_rng(seed),
        )

    sample["rResid"] = r_resid
    sample["rObserved"] = np.exp(r_resid + y_hat)

    elist["Sample"] = sample
    return elist

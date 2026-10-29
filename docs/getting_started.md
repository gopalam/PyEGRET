# Getting started with PyEGRET

This page explains how PyEGRET holds a record, what data it needs and in
what form, and the order in which an analysis usually runs. It assumes you
know what WRTDS is for; for the method itself, the EGRET user guide (Hirsch
and De Cicco, 2015, <https://doi.org/10.3133/tm4A10>) is the reference.

## The eList: one site, one constituent

Everything about one site and one constituent travels together in an
**eList**, which in PyEGRET is an ordinary Python dictionary with the same
element names as R's eList:

| key | contents |
|---|---|
| `"INFO"` | a one-row DataFrame of site and parameter metadata |
| `"Daily"` | a DataFrame with one row per day of discharge record |
| `"Sample"` | a DataFrame with one row per water-quality sample |
| `"surfaces"` | the fitted WRTDS surface, a NumPy array, once a model is fitted |
| `"vector_year"`, `"vector_logQ"` | the year and log-discharge axes of `surfaces` |

You build one with `merge_report(INFO, Daily, Sample)` and read it like any
dictionary: `elist["Daily"]`. Functions take an eList and **return a new
one**; the eList you pass in is never changed, so you can keep the
before-and-after side by side:

```python
fitted = model_estimation(elist)      # elist itself is unchanged
```

`describe_elist(elist)` returns a short summary to print.

## The data PyEGRET needs

**Daily discharge.** A continuous daily record, at least as long as the
sample record and preferably longer. `Daily` holds, among others, `Date`,
`Q` (m3/s), `LogQ`, `DecYear` (decimal year), `Month`, `Day` (day of the
year, leap-adjusted as in EGRET), `MonthSeq`, `waterYear`, and the 7- and
30-day means `Q7` and `Q30`. The readers build all of these for you.

**Water-quality samples.** Typically a few hundred samples over twenty years
or more; WRTDS needs about 100 samples within each estimation window by
default. `Sample` holds `Date`, the censoring bounds `ConcLow` and
`ConcHigh` (mg/L), `Uncen` (1 for a measured value, 0 for a censored one),
`ConcAve`, and the discharge of the day, `Q` and `LogQ`.

**Site information.** `INFO` needs at least `shortName` and `paramShortName`,
which label tables and plots. `drainSqKm` (drainage area) is used by a few
outputs. `set_pa` adds the period of analysis (below).

### Censored values

A value reported below a laboratory reporting limit ("<0.05") is
**left-censored**: the true concentration lies somewhere between zero and the
limit. PyEGRET stores it as `ConcLow = 0`, `ConcHigh = 0.05`, `Uncen = 0`
(R writes `NA` where PyEGRET writes 0; the two mean the same). A value above
an upper limit (">50") is right-censored: `ConcLow = 50`, `ConcHigh = NaN`.
The regression uses the full interval likelihood, exactly as R EGRET's
`survreg(Surv(..., type = "interval2"))` does, so censored values are used,
never dropped or replaced by half the limit.

### Units

Internally, as in EGRET, discharge is in m3/s and flux in kg/day. Tables and
plots convert for display with EGRET's unit codes:

- discharge, `q_unit`: 1 cfs, 2 m3/s, 3 thousand cfs, 4 thousand m3/s
- flux, `flux_unit`: 1 lbs/day, 2 tons/day, 3 kg/day, 4 thousand kg/day,
  5 tons/yr, ..., 8 thousand kg/yr, 9 million kg/yr, 13 kg/yr

Names work too (`flux_unit="kgDay"`); `pyegret.utils.print_flux_unit_cheat_sheet()` lists
them all. Concentration is assumed to be in mg/L; flux is computed from it,
so other concentration units give wrong fluxes.

## Getting data in

### From your own files

```python
from pyegret.data.read_user import read_user_daily, read_user_sample

daily = read_user_daily("daily.csv")              # columns: date, discharge
sample = read_user_sample("sample.csv")           # columns: date, remark, value
```

As in R, **`read_user_daily` assumes discharge in cubic feet per second**
and converts it; pass `q_unit=2` if your file is already in m3/s. Dates may
be `2001-03-15` or `03/15/2001`. In the sample file the remark is blank for a
measured value, `<` or `>` for a censored one. Several remark/value pairs in
one row (nitrate and nitrite reported separately, say) are summed into one
concentration, as EGRET does. Days with no discharge are dropped, and zero
flows are shifted slightly upward so their logarithm exists, again as in R.

### From USGS

With the optional `dataretrieval` package installed (`pip install
dataretrieval`), the readers use the USGS Water Data API:

```python
from pyegret.data.read_waterdata import (
    read_waterdata_daily, read_waterdata_sample, read_waterdata_info)

site = "01491000"                                  # Choptank River near Greensboro, MD
daily = read_waterdata_daily(site, "00060", "1979-10-01", "2011-09-30")
sample = read_waterdata_sample(site, "00631", "1979-10-01", "2011-09-30")
info = read_waterdata_info(site, "00631")
```

These translate R's `readNWISDaily`, `readNWISSample` and `readNWISInfo`,
which in current EGRET also call the new USGS services. As in R,
`read_waterdata_sample` keeps every result it receives, including several on
one day, and carries descriptive columns (activity type, medium, start time)
so that you can decide what to keep; it prints a note when those vary.
[Tutorial 3](../tutorials/03_usgs_data.ipynb) walks through a download and
those decisions. The readers were checked against the live service in October
2026: the Choptank River download reproduces the record distributed with
EGRET. The older `read_nwis_*` functions use the legacy NWIS services, which
USGS is retiring.

### Then join them

```python
from pyegret import merge_report
from pyegret.data.quality import data_overview

elist = merge_report(info, daily, sample)
data_overview(elist)            # counts, censoring, and problems worth fixing first
```

`merge_report` attaches each sample's daily discharge by matching dates
exactly, and warns about samples with no discharge on their day.

## The period of analysis

Annual results are computed for a "period of analysis": by default the
water year (October through September), labelled by the year it ends in.
For a season, say April through August:

```python
from pyegret.analysis import set_pa
elist = set_pa(elist, pa_start=4, pa_long=5)
```

Tables, plots and trend functions read the period from `INFO` from then on.
A year is reported only if the daily record covers all of it.

## A typical analysis

```python
from pyegret import model_estimation, wrtds_kalman, blank_time
from pyegret.analysis import (error_stats, flux_bias_stat, table_results,
                              table_change, run_pairs, run_groups, run_series)
import pyegret.plots as plots

# 1. look at the data
fig, axes = plots.multi_plot_data_overview(elist)

# 2. fit the model, then check it
elist = model_estimation(elist)
error_stats(elist)
flux_bias_stat(elist["Sample"])          # bias3 near 0 is good
fig, axes = plots.flux_bias_multi(elist)

# 3. results
table_results(elist)
table_change(elist, year_points=[1990, 2000, 2010])
fig, ax = plots.plot_conc_hist(elist)
fig, ax = plots.plot_contours(elist, 1990, 2020, q_bottom=0.5, q_top=50)

# 4. trends, with generalized flow normalization
pairs = run_pairs(elist, 1995, 2015, window_side=7)
pairs["pairResults"]                     # total change, CQTC, QTC
series = run_series(elist, window_side=7)

# 5. best daily estimates
kalman = wrtds_kalman(elist, seed=1)
fig, axes = plots.plot_wrtds_kalman(kalman)
```

A few things worth knowing:

- **Plots** return `(fig, ax)` and never call `plt.show()`; call it
  yourself, or `fig.savefig("figure.png", dpi=300)`. Pass `ax=` to draw
  into a figure of your own.
- **`window_side` has no default** in `run_pairs`, `run_groups` and
  `run_series`, as in R. `0` is classic flow normalization against the whole
  record; EGRET's guidance suggests 7.
- **Long gaps in sampling**: WRTDS will interpolate across them, which can
  look more certain than it is. `blank_time(elist, "2001-10-01",
  "2011-09-30")` removes the estimates for such a stretch.
- **Significance**: a change reported by `run_pairs` or `run_groups` is an
  estimate. Its uncertainty comes from the bootstrap, below.

## Uncertainty: confidence intervals and p-values

`pyegret.ci` translates EGRETci, the companion package to EGRET (Hirsch,
Archfield and De Cicco, 2015). It refits the model to many block-bootstrap
resamples of the samples (blocks of 200 days keep their serial correlation)
and uses the spread of the results:

```python
from pyegret.ci import run_pairs_boot, plot_histogram_trend, ci_calculations
from pyegret.ci import plot_conc_hist_boot

pairs = run_pairs(elist, 1995, 2015, window_side=7)
boot = run_pairs_boot(elist, pairs, n_boot=100)   # a few minutes
boot["bootOut"]          # 90 % interval (lowC, upC), p-value (pValC), likelihoods
boot["wordsOut"]         # e.g. "Upward trend in concentration is very likely"
fig, ax = plot_histogram_trend(elist, boot, flux=False)

band = ci_calculations(elist, n_boot=100)          # most of an hour
fig, ax = plot_conc_hist_boot(elist, band)
```

`run_groups_boot` does the same for `run_groups`, and `gen_daily_boot` with
`make_annual_pi` gives prediction intervals for WRTDS-Kalman loads. Each
bootstrap replicate is drawn exactly as EGRETci draws it, so results can be
compared with R's directly; set `start_seed` to vary them. For publication,
EGRETci's authors suggest at least 100 replicates.

## Finding a function

[r_to_python.md](r_to_python.md) lists every R function with its Python
name and module. Every function also has a docstring saying which R function
it mirrors: `help(pyegret.analysis.run_pairs)`.

# Instructions for working on PyEGRET

Read this before changing code. `docs/dev/ASSESSMENT.md` says what is done and what is
left; this file says how the code must be written. (The original eight-phase
build plan this file used to hold is in git history, before October 2026; it
described an `EList` class, which is gone.) The package was renamed from
`egret` to `pyegret` for v0.1.0. `CONTRIBUTING.md` at the top level gives
the same rules in short form for outside contributors.

PyEGRET is a Python translation of the USGS R packages **EGRET** and
**EGRETci**. The R sources, unmodified, are in `reference/EGRET_R/` and
`reference/EGRETci_R/`. Translate from them, function by function.

## Non-negotiable style rules

1. **No object-oriented code.** No classes, no dataclasses, no methods, no
   properties - in `pyegret/` *and* in `tests/`. Functions only.
2. **The eList is a plain dict** with R's keys: `"INFO"`, `"Daily"`,
   `"Sample"`, `"surfaces"`, plus `"vector_year"` and `"vector_logQ"` (the
   grid axes R keeps as attributes of `surfaces`). Build one with
   `pyegret.elist.make_elist`. Where R returns a list or a data frame with
   attributes (`runPairs`, `runGroups`), return a dict with the same names
   (`"pairResults"`, `"dateInfo"`, `"Other"`, ...).
3. **No chained calls.** Never call a method on the result of another call:
   not `df.sort_values("Date").reset_index(drop=True)`, not
   `pd.to_numeric(x).to_numpy()`, not `np.isnan(x).sum()`. Write one step
   per line and assign it, or use a NumPy function (`np.sum(np.isnan(x))`).
   `pyegret/utils/convert.py` has `as_float_array`, `as_dates` and
   `clean_text` for the conversions that would otherwise be chains.
   Check with the AST script in `diagnostics/check_style.py`; it must report 0.
4. **Works without being installed.** `sys.path.insert(0, r"...\PyEGRET")`
   then `import pyegret` must always work. `pyproject.toml` exists only as
   metadata so that others can `pip install` the package; it must never add
   build steps, compiled code or anything the plain-folder route lacks.
   Scripts in `diagnostics/` and `examples/` put the project folder on
   `sys.path` themselves; `pytest` finds it through `conftest.py`.
   `diagnostics/check_style.py` checks this too.
5. **Do not change the caller's eList.** A function that returns an eList
   starts with `elist = dict(elist)` and copies any DataFrame it modifies.
   (`est_surfaces` and `stitch_surfaces` still write the grid axes into the
   dict they are given; that is documented and is the only exception.)
6. Simple, explicit code: plain loops are fine; no lambdas where a named
   function reads better; no clever comprehensions over three lines.

## Translation conventions

- **Names**: R camelCase becomes snake_case (`runPairs` -> `run_pairs`,
  `windowSide` -> `window_side`). DataFrame **column names stay exactly as in
  R** (`ConcDay`, `FNConc`, `paStart`), so R knowledge transfers.
- **Arguments**: same order and defaults as R. If R has no default (as
  `windowSide` in `runPairs`), Python has none either.
- **Units and numbering**: discharge in m3/s and flux in kg/day internally;
  `qUnit` / `fluxUnit` use EGRET's numbering (`pyegret/utils/units.py`), and
  accept the EGRET names too.
- **Docstrings** say which R function is mirrored and anything that differs.
  If R has a quirk, reproduce it and say so (see `surface_start_end`,
  `run_groups` group membership); if R has a plain bug, fix it and say so
  (see `start_end`).
- **Censoring**: PyEGRET writes `ConcLow = 0` for a left-censored sample
  where R writes NA; both mean the same and every function accepts either.
  A right-censored sample has `ConcHigh = NaN`, as in R.
- **Surfaces** are `[year, logQ, slice]`; R's are `[logQ, year, slice]`.
  Transpose when comparing.
- **Plots** return `(fig, ax)` (or `(fig, axes)`), accept `ax=None`, and
  never call `plt.show()`.

## Validation, which every change must keep passing

```
python -m pytest                                  # all tests, offline
python diagnostics/validate_against_egret.py      # Choptank surface vs R
python diagnostics/check_style.py                 # no classes, no chains
```

`tests/reference/` holds results computed by R EGRET itself (see its
README): a fitted Choptank eList, a Chain Bridge eList fitted with current
defaults, two `runPairs` results and a `runSeries` result. New ports should
be checked against R output wherever R output can be had; synthetic data
with a known answer (`pyegret.testing.make_synthetic_elist`) is the fallback.

## Working on the user's machine

Work on a branch; commit
in small steps with messages that say what changed and why.

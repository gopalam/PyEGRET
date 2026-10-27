# Contributing to PyEGRET

Thank you for your interest. Bug reports, comparisons with R EGRET on your
own data, and translations of functions not yet ported are all welcome.

## Reporting a problem

Open an issue on GitHub (<https://github.com/gopalam/PyEGRET/issues>) with the PyEGRET version (`pyegret.__version__`),
what you ran, what you expected, and, where you can, what R EGRET gives for
the same data. A difference from R is the most useful kind of report, since
matching R is the project's aim.

## How the code is written

PyEGRET is written to be read by scientists who know R or MATLAB better than
Python's object model, so it keeps to a few rules:

1. **Functions only.** No classes, methods or dataclasses. The eList is a
   plain dict with R's element names.
2. **No chained calls.** One call per line, assigned to a name:
   `daily = daily.sort_values("Date")` then
   `daily = daily.reset_index(drop=True)`, not the two chained.
3. **It must work without being installed**: put the folder on `sys.path`
   and `import pyegret`.
4. **Follow R.** Each function translates an EGRET function: same arguments,
   in the same order, with the same defaults, in snake_case; same column
   names; and a docstring that says which R function it mirrors. The R
   sources are in `reference/`. Where R has a quirk, reproduce it and say so
   in the docstring; where R has a plain bug, fix it and say so.
5. **Do not change the caller's eList.** Return a new one.

`python diagnostics/check_style.py` checks rules 1 to 3. The fuller
developer notes are in `docs/dev/`.

## Before you open a pull request

```
python -m pytest
python diagnostics/check_style.py
```

Both must pass. If you changed anything the tutorials use, also run
`python diagnostics/run_tutorials.py` (it needs `pip install nbclient
ipykernel`); add `--save` to store the new results in the notebooks. New
translations need tests, and wherever R output can be
had, a comparison with it (see `tests/test_egret_trend_reference.py`).

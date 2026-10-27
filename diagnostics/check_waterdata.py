"""
Verify the modern USGS Water Data API retrieval. Run this on a machine with
ordinary internet access.

The USGS endpoints are unreachable from the sandboxes this package was
developed in, so ``pyegret/data/read_waterdata.py`` is the one module written
without being exercised against the live service. This script exercises it and
prints everything needed to fix a mismatch in one round trip.

The strong check is at the end: it fetches the Choptank River record from the
live API and compares it against the same record as EGRET shipped it in
``tests/reference/Choptank_eList.RData``. Same site, same parameter, same
years. If the discharge values agree, retrieval and unit conversion are both
verified against known-good data.

Run:  python diagnostics/check_waterdata.py
Needs: pip install "dataretrieval>=1.3" rdata
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

SITE = "01491000"          # Choptank River at Greensboro, MD
PARAM = "00631"            # nitrate plus nitrite
START, END = "1979-10-01", "2011-09-30"

_rule = "=" * 72


def _heading(text: str) -> None:
    print(f"\n{_rule}\n  {text}\n{_rule}")


def _describe(frame: pd.DataFrame, label: str) -> None:
    print(f"{label}: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"  columns: {list(frame.columns)}")
    if not frame.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            first_rows = frame.head(3)
            print(first_rows.to_string())


def environment() -> None:
    _heading("Environment")
    print(f"python       {sys.version.split()[0]}   ({sys.executable})")
    for name in ("numpy", "pandas", "scipy", "dataretrieval", "rdata"):
        try:
            mod = __import__(name)
            print(f"{name:12} {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            print(f"{name:12} NOT INSTALLED")
    if sys.version_info < (3, 9):
        print("\n  !! Python 3.9 or newer is required (numpy 2.x drops 3.8).")


def raw_responses() -> dict:
    """Fetch each endpoint untouched and show exactly what comes back."""
    from pyegret.data import read_waterdata as rw

    out = {}
    for label, call in [
        ("daily", lambda: rw.read_waterdata_daily(SITE, "00060", START, END, raw=True)),
        ("samples", lambda: rw.read_waterdata_sample(SITE, PARAM, START, END, raw=True)),
        ("info", lambda: rw.read_waterdata_info(SITE, PARAM, raw=True)),
    ]:
        _heading(f"Raw response: {label}")
        try:
            frame = call()
            out[label] = frame
            _describe(frame, label)
        except Exception:
            print(f"FAILED:\n{traceback.format_exc()}")
    return out


def converted() -> dict:
    """Run the EGRET-format conversion and sanity-check the result."""
    from pyegret.data import read_waterdata as rw

    out = {}
    _heading("Converted to EGRET format")

    try:
        daily = rw.read_waterdata_daily(SITE, "00060", START, END)
        out["daily"] = daily
        _describe(daily, "Daily")
        first_day = pd.Timestamp(daily["Date"].min())
        last_day = pd.Timestamp(daily["Date"].max())
        print(f"  date range   {first_day:%Y-%m-%d} .. {last_day:%Y-%m-%d}")
        print(f"  Q (m3/s)     min {daily['Q'].min():.4f}  median "
              f"{daily['Q'].median():.4f}  max {daily['Q'].max():.2f}")
        if daily["Q"].median() > 50:
            print("  !! Median discharge looks like ft3/s, not m3/s - check the unit conversion.")
    except Exception:
        print(f"Daily FAILED:\n{traceback.format_exc()}")

    try:
        sample = rw.read_waterdata_sample(SITE, PARAM, START, END)
        out["sample"] = sample
        _describe(sample, "Sample")
        print(f"  censored     {int(np.sum((sample['Uncen'] == 0)))} of {len(sample)}")
        print(f"  ConcHigh     min {sample['ConcHigh'].min():.4f}  max "
              f"{sample['ConcHigh'].max():.4f}")
    except Exception:
        print(f"Sample FAILED:\n{traceback.format_exc()}")

    try:
        info = rw.read_waterdata_info(SITE, PARAM)
        out["info"] = info
        print(f"\nINFO shortName   {info['shortName'].iloc[0]!r}")
        print(f"INFO drainSqKm   {info['drainSqKm'].iloc[0]}   (EGRET reports 292.67)")
    except Exception:
        print(f"INFO FAILED:\n{traceback.format_exc()}")

    return out


def compare_with_reference(fetched: dict) -> None:
    """Compare the live fetch against EGRET's own copy of the same record."""
    _heading("Live fetch vs EGRET's shipped Choptank record")

    path = os.path.join(_ROOT, "tests", "reference", "Choptank_eList.RData")
    if not os.path.exists(path):
        print(f"reference not found at {path} - skipping")
        return
    try:
        import rdata
    except ImportError:
        print("rdata not installed (pip install rdata) - skipping")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elist = rdata.conversion.convert(rdata.parser.parse_file(path))["Choptank_eList"]
    daily_r = elist["Daily"]
    sample_r = elist["Sample"]
    for frame in (daily_r, sample_r):
        frame.columns = [str(c) for c in frame.columns]

    daily = fetched.get("daily")
    if daily is not None:
        merged = daily.merge(
            daily_r[["Date", "Q"]].rename(columns={"Q": "Q_egret"}),
            on="Date", how="inner",
        )
        print(f"Daily: {len(merged)} dates in common "
              f"(live {len(daily)}, EGRET {len(daily_r)})")
        if len(merged):
            rel = np.abs((merged["Q"] - merged["Q_egret"]) / merged["Q_egret"])
            print(f"  discharge agreement: median {np.median(rel):.2e}  "
                  f"99th pct {np.percentile(rel, 99):.2e}  worst {rel.max():.2e}")
            if np.median(rel) < 1e-6:
                print("  PASS - discharge matches EGRET, so units and parsing are right.")
            elif np.median(rel) < 0.02:
                print("  Close. Small differences are plausible if USGS has since "
                      "revised the record.")
            else:
                ratio = float((merged["Q"] / merged["Q_egret"]).median())
                cfs_to_cms = 0.028316847
                print(f"  MISMATCH - median ratio live/EGRET is {ratio:.5f}.")
                print(f"     A ratio near {1 / cfs_to_cms:.1f} means the ft3/s to m3/s "
                      f"conversion was skipped;")
                print(f"     a ratio near {cfs_to_cms:.5f} means it was applied twice.")

    sample = fetched.get("sample")
    if sample is not None:
        merged = sample.merge(
            sample_r[["Date", "ConcHigh"]].rename(columns={"ConcHigh": "Conc_egret"}),
            on="Date", how="inner",
        )
        print(f"\nSample: {len(merged)} dates in common "
              f"(live {len(sample)}, EGRET {len(sample_r)})")
        if len(merged):
            rel = np.abs((merged["ConcHigh"] - merged["Conc_egret"])
                         / merged["Conc_egret"].replace(0, np.nan))
            print(f"  concentration agreement: median {np.nanmedian(rel):.2e}  "
                  f"worst {np.nanmax(rel):.2e}")


def main() -> None:
    environment()
    raw_responses()
    fetched = converted()
    compare_with_reference(fetched)
    _heading("Done")
    print("Paste this whole output back and any column mismatch can be fixed\n"
          "without another round trip.")


if __name__ == "__main__":
    main()

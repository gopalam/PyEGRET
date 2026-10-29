"""
Building blocks of EGRETci's bootstrap: block resampling, the two-sided
p-value, the likelihood wording, and the trend set-up of the original WRTDS
Bootstrap Test.

Mirrors R EGRETci 2.0.5 ``blockSample()``, ``pVal()``, ``wordLike()``,
``words()``, ``trendSetUp()`` and ``setForBoot()``, and the
``binom::binom.bayes()`` interval that ``wBT()`` uses to decide when to stop.

The method is described in Hirsch, R.M., Archfield, S.A., and De Cicco, L.A.,
2015, A bootstrap method for estimating uncertainty of water quality trends:
Environmental Modelling & Software, v. 73, p. 148-166,
https://doi.org/10.1016/j.envsoft.2015.07.017.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pyegret.ci.r_random import r_sample_index, r_set_seed

__all__ = [
    "block_sample",
    "p_val",
    "word_like",
    "words",
    "binom_bayes",
    "trend_set_up",
    "set_for_boot",
    "DEFAULT_START_SEED",
]

# EGRETci's default seed.
DEFAULT_START_SEED = 494817


def random_seed() -> int:
    """A seed drawn at random, for when none is given."""
    generator = np.random.default_rng()
    return int(generator.integers(1, 2**31 - 1))


def block_sample_with_state(sample: pd.DataFrame, block_length: int,
                            rng_state: dict) -> tuple:
    """:func:`block_sample` drawing from a given R generator state.

    Returns:
        ``(resampled Sample, generator state afterwards)``.
    """
    julian = sample["Julian"].to_numpy(dtype=float)
    n_samples = len(sample)
    day_one = julian[0]
    first_julian = julian[0] - block_length + 1
    last_julian = julian[-1]
    n_starts = int(last_julian - first_julian) + 1

    blocks = []
    n_rows = 0
    state = rng_state
    while n_rows <= n_samples:
        index, state = r_sample_index(state, n_starts)
        random_date = first_julian + index
        block_start = max(random_date, day_one)
        block_end = min(last_julian, random_date + block_length - 1)
        rows = np.flatnonzero((julian >= block_start) & (julian < block_end))
        # R: newSample <- rbind(oneYear, newSample), so each block goes first.
        blocks.insert(0, rows)
        n_rows = n_rows + len(rows)

    chosen = np.concatenate(blocks)
    chosen = chosen[:n_samples]
    # R's order() is stable, so repeated samples keep their drawn order.
    order = np.argsort(julian[chosen], kind="stable")
    chosen = chosen[order]
    resampled = sample.iloc[chosen]
    resampled = resampled.reset_index(drop=True)
    return resampled, state


def block_sample(sample: pd.DataFrame, block_length: int = 200,
                 start_seed: int | None = None) -> pd.DataFrame:
    """A block-bootstrap resample of the Sample. Mirrors R ``blockSample()``.

    Blocks of ``block_length`` days are placed at random over the record and
    their samples copied into the new set until it has as many samples as the
    original; the excess is trimmed and the result sorted by date. Keeping
    whole blocks preserves the serial correlation of the data.

    With ``start_seed`` the draw is exactly R's: ``RNGversion("3.5.0");
    set.seed(start_seed)`` followed by ``sample()``, so the same rows are
    chosen as in EGRETci.

    Args:
        sample: Sample DataFrame, sorted by date, with a ``Julian`` column.
        block_length: Block length in days. Default 200, as in EGRETci.
        start_seed: Seed. None draws a seed at random.

    Returns:
        The resampled Sample, same columns, ``len(sample)`` rows.
    """
    if start_seed is None:
        start_seed = random_seed()
    resampled, _ = block_sample_with_state(sample, block_length, r_set_seed(start_seed))
    return resampled


def p_val(s) -> float:
    """Two-sided p-value from bootstrap trend replicates. Mirrors R ``pVal()``.

    Zero and missing replicates are ignored. The p-value comes from where
    zero falls between the largest negative and the smallest positive
    replicate, interpolated. When every replicate has the same sign it is
    ``2 / (m + 1)``, which should be reported as "less than" that value.
    """
    values = np.asarray(s, dtype=float)
    values = values[~np.isnan(values)]
    values = values[np.abs(values) > 0]
    values = np.sort(values)
    m = len(values)
    x = int(np.sum(values > 0))
    if x == 0 or x == m:
        return 2.0 / (m + 1)
    kp = m - x + 1          # order statistic of the smallest positive (1-based)
    kn = kp - 1             # order statistic of the largest negative
    b1 = (kn - kp) / (values[kn - 1] - values[kp - 1])
    k0 = kp - (b1 * values[kp - 1])
    p = k0 / (m + 1)
    return float(2.0 * min(p, 1.0 - p))


_LIKE_SUBJECTS = (
    "Upward trend in concentration is",
    "Downward trend in concentration is",
    "Upward trend in flux is",
    "Downward trend in flux is",
)
_LIKE_WORDS = (
    "highly unlikely",
    "very unlikely",
    "unlikely",
    "about as likely as not",
    "likely",
    "very likely",
    "highly likely",
)
_LIKE_BREAKS = np.array([0.0, 0.05, 0.1, 0.33, 0.67, 0.9, 0.95, 1.0])


def word_like(like_list) -> list:
    """Four sentences describing the likelihoods of trends. Mirrors R ``wordLike()``.

    Args:
        like_list: Likelihoods of an upward and a downward trend in
            concentration, then in flux.

    Returns:
        Four strings such as ``"Upward trend in flux is highly likely"``.
    """
    sentences = []
    for subject, likelihood in zip(_LIKE_SUBJECTS, like_list):
        # R's cut() with right-closed intervals (a, b].
        level = int(np.searchsorted(_LIKE_BREAKS, likelihood, side="left"))
        if level < 1 or level > len(_LIKE_WORDS):
            sentences.append(f"{subject} NA")
        else:
            sentences.append(f"{subject} {_LIKE_WORDS[level - 1]}")
    return sentences


def words(reject: bool) -> str:
    """``"Reject Ho"`` or ``"Do Not Reject Ho"``. Mirrors R ``words()``."""
    if reject:
        return "Reject Ho"
    return "Do Not Reject Ho"


def binom_bayes(x: int, n: int, conf_level: float) -> dict:
    """Bayesian binomial interval, Jeffreys prior, central type.

    Mirrors R ``binom::binom.bayes(x, n, conf.level, type = "central")``: the
    Beta(x + 0.5, n - x + 0.5) posterior, its mean, and its central interval;
    at x = 0 or x = n the interval is one-sided and closed at 0 or 1.

    Returns:
        Dict with ``mean``, ``lower`` and ``upper``.
    """
    a = x + 0.5
    b = n - x + 0.5
    ends = (x == 0) or (x == n)
    alpha = 1.0 - conf_level
    if not ends:
        alpha = alpha * 0.5
    lower = float(stats.beta.ppf(alpha, a, b))
    upper = float(stats.beta.ppf(1.0 - alpha, a, b))
    if x == 0:
        lower = 0.0
    if x == n:
        upper = 1.0
    return {"mean": a / (a + b), "lower": lower, "upper": upper}


def trend_set_up(elist: dict, year1: int, year2: int, n_boot: int = 100,
                 boot_break: int = 39, block_length: int = 200) -> pd.DataFrame:
    """The set-up for :func:`~pyegret.ci.wbt.wbt`. Mirrors R ``trendSetUp()``.

    R asks for any value not given; here all have defaults (``boot_break`` 39
    follows R's prompt, "values of 39 or greater produce more accurate CIs").

    Args:
        elist: eList dict with Sample.
        year1, year2: First and last years of the trend period.
        n_boot: Maximum number of bootstrap replicates.
        boot_break: Minimum number of replicates (capped at ``n_boot``).
        block_length: Bootstrap block length in days.

    Returns:
        One-row DataFrame with ``year1``, ``yearData1``, ``year2``,
        ``yearData2``, ``numSamples``, ``nBoot``, ``bootBreak``,
        ``blockLength`` and ``confStop`` (0.7).

    Raises:
        ValueError: If a year lies outside the sample record.
    """
    dec_year = elist["Sample"]["DecYear"].to_numpy(dtype=float)
    num_samples = len(dec_year)
    if year1 < np.floor(np.nanmin(dec_year)):
        raise ValueError("year1 is less than the first Sample year")
    if year2 > np.ceil(np.nanmax(dec_year)):
        raise ValueError("year2 is greater than the last Sample year")
    boot_break = min(boot_break, n_boot)
    return pd.DataFrame([{
        "year1": year1,
        "yearData1": int(np.trunc(dec_year[0] + 0.25)),
        "year2": year2,
        "yearData2": int(np.trunc(dec_year[-1] + 0.25)),
        "numSamples": num_samples,
        "nBoot": n_boot,
        "bootBreak": boot_break,
        "blockLength": block_length,
        "confStop": 0.7,
    }])


def set_for_boot(elist: dict, window_y: float = 7, window_q: float = 2,
                 window_s: float = 0.5, edge_adjust: bool = True) -> dict:
    """Put the model settings into INFO for :func:`~pyegret.ci.wbt.wbt`.

    Mirrors R ``setForBoot()``: windows and minimums already in INFO are kept,
    missing ones get these defaults (minimums ``min(100, n - 20)``), and the
    record's dates are added. The grid is not stored; PyEGRET builds it from
    Daily when needed.

    Returns:
        A new eList; the one passed in is not changed.
    """
    elist = dict(elist)
    info = elist["INFO"].copy()
    daily = elist["Daily"]
    n_samples = len(elist["Sample"])

    defaults = {"windowY": window_y, "windowQ": window_q, "windowS": window_s,
                "minNumObs": min(100, n_samples - 20),
                "minNumUncen": min(100, n_samples - 20)}
    for name, value in defaults.items():
        if name not in info.columns or pd.isna(info[name].iloc[0]):
            info[name] = value
    info["edgeAdjust"] = edge_adjust
    dec_year = daily["DecYear"].to_numpy(dtype=float)
    info["numDays"] = len(dec_year)
    info["DecLow"] = dec_year[0]
    info["DecHigh"] = dec_year[-1]
    elist["INFO"] = info
    return elist

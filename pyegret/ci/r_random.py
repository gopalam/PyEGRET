"""
R's default random-number generator, so that bootstrap samples match EGRETci's.

EGRETci draws its block-bootstrap samples with ``set.seed(startSeed + iBoot)``
after ``RNGversion("3.5.0")``: R's Mersenne-Twister, seeded R's way, with the
"Rounding" method of ``sample()``. This module reproduces that sequence
exactly (R's ``src/main/RNG.c`` and ``unique.c``), so PyEGRET's bootstrap
replicate *i* resamples the same rows as EGRETci's replicate *i*.

The state is a plain dict, ``{"mt": [624 unsigned 32-bit ints], "mti": int}``;
each function takes a state and returns the values it drew and the new state.
"""
from __future__ import annotations

import math

__all__ = ["r_set_seed", "r_runif", "r_sample_index"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_MASK32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10     # 1 / (2^32 - 1)


def r_set_seed(seed: int) -> dict:
    """The generator state after R's ``set.seed(seed)`` (Mersenne-Twister).

    Mirrors R's ``RNG_Init``: 50 rounds of ``seed = 69069 * seed + 1`` to
    scramble, then the same recurrence fills the 625 seed words, of which the
    first becomes the position counter (set to 624, so the first draw
    regenerates the table).
    """
    value = int(seed) & _MASK32
    for _ in range(50):
        value = (69069 * value + 1) & _MASK32
    words = []
    for _ in range(_N + 1):
        value = (69069 * value + 1) & _MASK32
        words.append(value)
    return {"mt": words[1:], "mti": _N}


def _twist(mt: list) -> list:
    """Regenerate the 624-word table, as R's ``MT_genrand`` does."""
    mt = list(mt)
    for kk in range(_N):
        y = (mt[kk] & _UPPER_MASK) | (mt[(kk + 1) % _N] & _LOWER_MASK)
        value = mt[(kk + _M) % _N] ^ (y >> 1)
        if y & 1:
            value = value ^ _MATRIX_A
        mt[kk] = value
    return mt


def r_runif(state: dict, n: int = 1) -> tuple:
    """``n`` uniform numbers as R's ``runif(n)`` gives them, and the new state."""
    mt = state["mt"]
    mti = state["mti"]
    values = []
    for _ in range(n):
        if mti >= _N:
            mt = _twist(mt)
            mti = 0
        y = mt[mti]
        mti = mti + 1
        y = y ^ (y >> 11)
        y = y ^ ((y << 7) & 0x9D2C5680)
        y = y ^ ((y << 15) & 0xEFC60000)
        y = y ^ (y >> 18)
        value = (y & _MASK32) * 2.3283064365386963e-10
        # R's fixup(): keep the value strictly inside (0, 1).
        if value <= 0.0:
            value = 0.5 * _I2_32M1
        elif 1.0 - value <= 0.0:
            value = 1.0 - 0.5 * _I2_32M1
        values.append(value)
    return values, {"mt": mt, "mti": mti}


def r_sample_index(state: dict, n: int) -> tuple:
    """A zero-based index into ``n`` items, as R 3.5's ``sample(x, 1)`` chooses it.

    R before 3.6 (and ``RNGversion("3.5.0")``) uses the "Rounding" method:
    ``floor(n * unif_rand())``.
    """
    values, state = r_runif(state, 1)
    return int(math.floor(n * values[0])), state

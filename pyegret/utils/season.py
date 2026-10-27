"""
Period-of-analysis labels. Mirrors R ``setSeasonLabel()`` and
``setSeasonLabelByUser()``.
"""
from __future__ import annotations

__all__ = ["set_season_label", "set_season_label_by_user", "MONTH_ABBREV", "MONTH_FULL"]

MONTH_ABBREV = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
MONTH_FULL = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def set_season_label(paStart: int = 10, paLong: int = 12) -> str:
    """Return EGRET's label for a period of analysis.

    Mirrors R ``setSeasonLabel()``, which reads ``paStart`` / ``paLong`` from
    the annual results; here they are passed directly.

    Args:
        paStart: Starting month of the period (1-12). Default 10 (October).
        paLong: Length of the period in months (1-12). Default 12.

    Returns:
        ``"Water Year"`` for October + 12, ``"Calendar Year"`` for January + 12,
        ``"Year Starting With April"`` for any other 12-month period, and
        ``"Season Consisting of Apr May Jun"`` for a shorter season.
    """
    if paLong == 12 and paStart == 10:
        return "Water Year"
    if paLong == 12 and paStart == 1:
        return "Calendar Year"
    if paLong == 12:
        return "Year Starting With " + MONTH_FULL[paStart - 1]

    months = []
    for i in range(paLong):
        month = paStart + i
        if month > 12:
            month = month - 12
        months.append(MONTH_ABBREV[month - 1])
    return "Season Consisting of " + " ".join(months)


def set_season_label_by_user(paStartInput: int = 10, paLongInput: int = 12) -> str:
    """Same as :func:`set_season_label`. Mirrors R ``setSeasonLabelByUser()``."""
    return set_season_label(paStartInput, paLongInput)

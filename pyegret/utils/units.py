"""
Discharge and flux unit tables, copied from EGRET's ``qConst`` and ``fluxConst``
(R/objectDefinitions.R), with the same numbering.

PyEGRET stores discharge in m3/s and flux in kg/day internally, exactly as
EGRET does. A unit's ``factor`` converts from those internal units to the
display unit: ``value_in_unit = internal_value * factor``.

A unit can be chosen by its number or by its EGRET name, so R scripts carry
over unchanged: ``fluxUnit = 3`` and ``fluxUnit = "kgDay"`` both mean kg/day.

Flux units (EGRET numbering)::

     1 poundsDay        lbs/day
     2 tonsDay          tons/day (short tons)
     3 kgDay            kg/day
     4 thousandKgDay    10^3 kg/day
     5 tonsYear         tons/yr
     6 thousandTonsYear 10^3 tons/yr
     7 millionTonsYear  10^6 tons/yr
     8 thousandKgYear   10^3 kg/yr
     9 millionKgYear    10^6 kg/yr
    10 billionKgYear    10^9 kg/yr
    11 thousandTonsDay  10^3 tons/day
    12 millionKgDay     10^6 kg/day
    13 kgYear           kg/yr

Discharge units::

     1 cfs, 2 cms, 3 thousandCfs, 4 thousandCms, 5 mmDay
"""
from __future__ import annotations

__all__ = [
    "Q_UNITS",
    "FLUX_UNITS",
    "get_q_unit",
    "get_flux_unit",
    "print_q_unit_cheat_sheet",
    "print_flux_unit_cheat_sheet",
]


def _unit(name, short_name, factor, unit_name, label, tiny_label) -> dict:
    """One row of a unit table, as a dict."""
    entry = {
        "name": name,
        "short_name": short_name,
        "factor": factor,
        "unit_name": unit_name,
        "label": label,
        "tiny_label": tiny_label,
    }
    return entry


Q_UNITS: dict = {
    1: _unit("cfs", "cfs", 35.314667, "Cubic Feet per Second",
             "Discharge, in cubic feet per second", "Discharge (ft³/s)"),
    2: _unit("cms", "cms", 1.0, "Cubic Meters per Second",
             "Discharge, in cubic meters per second", "Discharge (m³/s)"),
    3: _unit("thousandCfs", "10^3 cfs", 0.035314667, "Thousand Cubic Feet per Second",
             "Discharge, in thousands of cubic feet per second",
             "Discharge (10³ ft³/s)"),
    4: _unit("thousandCms", "10^3 cms", 0.001, "Thousand Cubic Meters per Second",
             "Discharge, in thousands of cubic meters per second",
             "Discharge (10³ m³/s)"),
    5: _unit("mmDay", "mm/day", 86400000.0, "Cubic Millimeters per Day",
             "Discharge, in cubic millimeters per day", "Discharge (mm³/day)"),
}

FLUX_UNITS: dict = {
    1: _unit("poundsDay", "lbs/day", 2.204623, "pounds/day",
             "Flux, in pounds per day", "Flux (lbs/d)"),
    2: _unit("tonsDay", "tons/day", 0.001102, "tons/day",
             "Flux, in tons per day", "Flux (tons/d)"),
    3: _unit("kgDay", "kg/day", 1.0, "kg/day",
             "Flux, in kilograms per day", "Flux (kg/d)"),
    4: _unit("thousandKgDay", "10^3 kg/day", 0.001, "thousands of kg/day",
             "Flux, in thousands of kilograms per day", "Flux (10³ kg/d)"),
    5: _unit("tonsYear", "tons/yr", 0.402619, "tons/year",
             "Flux, in tons per year", "Flux (tons/yr)"),
    6: _unit("thousandTonsYear", "10^3 tons/yr", 0.000402619, "thousands of tons/year",
             "Flux, in thousands of tons per year", "Flux (10³ tons/yr)"),
    7: _unit("millionTonsYear", "10^6 tons/yr", 4.02619e-07, "millions of tons/year",
             "Flux, in millions of tons per year", "Flux (10⁶ tons/yr)"),
    8: _unit("thousandKgYear", "10^3 kg/yr", 0.36525, "thousands of kg/year",
             "Flux, in thousands of kilograms per year", "Flux (10³ kg/yr)"),
    9: _unit("millionKgYear", "10^6 kg/yr", 0.00036525, "millions of kg/year",
             "Flux, in millions of kilograms per year", "Flux (10⁶ kg/yr)"),
    10: _unit("billionKgYear", "10^9 kg/yr", 3.6525e-07, "billions of kg/year",
              "Flux, in billions of kilograms per year", "Flux (10⁹ kg/yr)"),
    11: _unit("thousandTonsDay", "10^3 tons/day", 1.102e-06, "thousands of tons/day",
              "Flux, in thousands of tons per day", "Flux (10³ tons/d)"),
    12: _unit("millionKgDay", "10^6 kg/day", 1.0e-06, "millions of kg/day",
              "Flux, in millions of kilograms per day", "Flux (10⁶ kg/d)"),
    13: _unit("kgYear", "kg/yr", 365.25, "kg/year",
              "Flux, in kilograms per year", "Flux (kg/yr)"),
}


def _lookup(table: dict, unit, table_name: str) -> dict:
    """Find a unit by number or by EGRET name."""
    if isinstance(unit, str):
        for entry in table.values():
            if entry["name"] == unit:
                return entry
        names = [entry["name"] for entry in table.values()]
        raise KeyError(f"{table_name} has no unit named {unit!r}. Valid names: {names}")
    if unit not in table:
        raise KeyError(f"{table_name} has no unit number {unit!r}. Valid: 1-{len(table)}")
    return table[unit]


def get_q_unit(unit) -> dict:
    """Return the discharge-unit entry for an EGRET qUnit number or name.

    Mirrors R ``qConst[[qUnit]]``.

    Args:
        unit: 1-5, or a name such as ``"cfs"``.

    Returns:
        Dict with keys ``name``, ``short_name``, ``factor``, ``unit_name``,
        ``label`` and ``tiny_label``.

    Raises:
        KeyError: If the unit is not in the table.
    """
    return _lookup(Q_UNITS, unit, "Q_UNITS")


def get_flux_unit(unit) -> dict:
    """Return the flux-unit entry for an EGRET fluxUnit number or name.

    Mirrors R ``fluxConst[[fluxUnit]]``.

    Args:
        unit: 1-13, or a name such as ``"kgDay"``.

    Returns:
        Dict with keys ``name``, ``short_name``, ``factor``, ``unit_name``,
        ``label`` and ``tiny_label``.

    Raises:
        KeyError: If the unit is not in the table.
    """
    return _lookup(FLUX_UNITS, unit, "FLUX_UNITS")


def print_q_unit_cheat_sheet() -> None:
    """Print the discharge units. Mirrors R ``printqUnitCheatSheet()``."""
    print("The following codes apply to the qUnit list:")
    for number, entry in Q_UNITS.items():
        print(f"{number} =  {entry['name']}  ( {entry['unit_name']} )")


def print_flux_unit_cheat_sheet() -> None:
    """Print the flux units. Mirrors R ``printFluxUnitCheatSheet()``."""
    print("The following codes apply to the fluxUnit list:")
    for number, entry in FLUX_UNITS.items():
        print(f"{number} =  {entry['name']}  ( {entry['unit_name']} )")

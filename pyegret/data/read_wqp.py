"""
Water-quality samples from the Water Quality Portal (WQP).

The Water Quality Portal (https://www.waterqualitydata.us) serves results from
state, tribal, county and federal agencies, which is where most non-USGS
monitoring data end up. Mirrors R ``readWQPSample()`` in EGRET 3.0.12.
"""
from __future__ import annotations

import warnings

import pandas as pd

from pyegret.data.qw_processing import wqp_results_to_sample

__all__ = ["read_wqp_sample"]


def _wqp_date(date: str, name: str) -> str:
    """``"YYYY-MM-DD"`` as the ``"MM-DD-YYYY"`` the Water Quality Portal expects."""
    parsed = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name} must be given as YYYY-MM-DD, not {date!r}.")
    return f"{parsed:%m-%d-%Y}"


def read_wqp_sample(
    site_id: str,
    characteristic_name: str,
    start_date: str = "",
    end_date: str = "",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Fetch water-quality results from the Water Quality Portal as an EGRET Sample.

    Mirrors R ``readWQPSample()``: the legacy ``resultPhysChem`` profile, then
    R's ``processQWData``, ``compressData``, ``remove_zeros`` and
    ``populateSampleColumns`` (see :mod:`pyegret.data.qw_processing`). A
    result reported as not detected becomes a "less than" value with its
    detection limit as the upper bound.

    As in R, a USGS site (an id containing ``USGS``) is passed on to
    :func:`~pyegret.data.read_waterdata.read_waterdata_sample`, because the
    portal's copy of USGS data stopped being updated in March 2024.

    As in R, every result is kept, and the descriptive columns
    (``CharacteristicName``, ``ResultSampleFraction``, ``ActivityTypeCode``,
    ``ActivityMediaName`` and others) are carried in the frame. One
    characteristic name often covers several fractions or methods (for
    example ``"Phosphorus"`` covers total and dissolved phosphorus); when
    those vary a message lists them, and you should keep only the ones you
    mean to model.

    Args:
        site_id: Water Quality Portal site identifier, an organization code
            and a site code joined by a hyphen (``"NOOKSACK_WQX-SW08"``).
        characteristic_name: Characteristic name, case-sensitive
            (``"Phosphorus"``, ``"Nitrate"``); see
            https://www.waterqualitydata.us/public_srsnames/.
        start_date: ``"YYYY-MM-DD"`` or ``""``.
        end_date: ``"YYYY-MM-DD"`` or ``""``.
        verbose: Print R's messages. Default True.
        **kwargs: Passed through to ``dataretrieval.wqp.get_results``.

    Returns:
        Sample DataFrame sorted by date, with ``Date``, ``ConcLow``,
        ``ConcHigh``, ``Uncen``, ``ConcAve``, ``LogConcLow``, ``LogConcHigh``,
        the date columns and the descriptive columns.

    Raises:
        ImportError: If ``dataretrieval`` is not installed.
        ValueError: If a date is not ``YYYY-MM-DD``, or nothing is returned.
    """
    site_id = str(site_id)
    if "USGS" in site_id:
        from pyegret.data.read_waterdata import read_waterdata_sample

        if verbose:
            print("Please use read_waterdata_sample for USGS data; calling it now.")
        return read_waterdata_sample(site_id, characteristic_name, start_date,
                                     end_date, verbose=verbose)

    try:
        from dataretrieval import wqp
    except ImportError as exc:
        raise ImportError(
            "dataretrieval is required for read_wqp_sample. "
            "Install with: pip install dataretrieval"
        ) from exc

    query = {
        "siteid": site_id,
        "characteristicName": characteristic_name,
        "dataProfile": "resultPhysChem",
    }
    if start_date:
        query["startDateLo"] = _wqp_date(start_date, "start_date")
    if end_date:
        query["startDateHi"] = _wqp_date(end_date, "end_date")
    query.update(kwargs)

    with warnings.catch_warnings():
        # The legacy profile is what R reads too; dataretrieval's warning about
        # it concerns USGS data, which are sent to read_waterdata_sample above.
        warnings.filterwarnings("ignore", message=".*legacy WQX format.*")
        result = wqp.get_results(legacy=True, **query)
    frame = result[0]
    if frame is None or len(frame) == 0:
        raise ValueError(
            f"No results returned for site {site_id!r}, characteristic "
            f"{characteristic_name!r}. Characteristic names are case-sensitive; "
            "see https://www.waterqualitydata.us/public_srsnames/."
        )
    frame = frame.reset_index(drop=True)
    return wqp_results_to_sample(frame, verbose=verbose)

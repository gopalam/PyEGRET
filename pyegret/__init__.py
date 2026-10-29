"""
PyEGRET: Python implementation of USGS EGRET
(Exploration and Graphics for RivEr Trends)

Mirrors: https://github.com/DOI-USGS/EGRET
"""
from pyegret.elist import (
    make_elist, has_sample, has_model, describe_elist,
    n_observations, n_censored_vals, n_discharge, blank_time,
)
from pyegret.data.merge_report import merge_report
from pyegret.model.model_estimation import model_estimation
from pyegret.model.kalman import wrtds_kalman

__version__ = "0.2.0"

__all__ = [
    "make_elist",
    "has_sample",
    "has_model",
    "describe_elist",
    "n_observations",
    "n_censored_vals",
    "n_discharge",
    "blank_time",
    "merge_report",
    "model_estimation",
    "wrtds_kalman",
]

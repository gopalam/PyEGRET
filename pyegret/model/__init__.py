"""Model subpackage: eList helpers and WRTDS statistical engine."""
from pyegret.elist import make_elist, has_sample, has_model, describe_elist
from pyegret.model.tri_cube import (
    tri_cube,
    season_distance,
    compute_weights,
    compute_adaptive_weights,
)
from pyegret.model.run_surv_reg import run_surv_reg
from pyegret.model.est_surfaces import est_surfaces, build_grid_axes
from pyegret.model.est_cross_val import est_cross_val
from pyegret.model.flex_fn import flex_fn, make_date_info
from pyegret.model.wall import stitch_surfaces, decimal_high_low
from pyegret.model.est_daily import est_daily_from_surfaces
from pyegret.model.model_estimation import model_estimation
from pyegret.model.kalman import wrtds_kalman

__all__ = [
    "make_elist",
    "has_sample",
    "has_model",
    "describe_elist",
    "tri_cube",
    "season_distance",
    "compute_weights",
    "compute_adaptive_weights",
    "run_surv_reg",
    "est_surfaces",
    "build_grid_axes",
    "est_cross_val",
    "flex_fn",
    "make_date_info",
    "stitch_surfaces",
    "decimal_high_low",
    "est_daily_from_surfaces",
    "model_estimation",
    "wrtds_kalman",
]

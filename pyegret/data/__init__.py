"""Data I/O subpackage: NWIS retrieval, user files, population, merging."""
from pyegret.data.populate import (
    populate_date_columns,
    populate_sample_columns,
    merge_daily_q,
)
from pyegret.data.merge_report import merge_report
from pyegret.data.compress_data import compress_data, remove_duplicates
from pyegret.data.read_user import read_user_daily, read_user_sample
from pyegret.data.quality import (
    populate_concentrations,
    process_qw_data,
    fix_sample_frame,
    data_overview,
    make_augmented_sample,
)
from pyegret.data.read_waterdata import (
    read_waterdata_daily,
    read_waterdata_sample,
    read_waterdata_info,
    usgs_location_id,
)

__all__ = [
    "populate_date_columns",
    "populate_sample_columns",
    "merge_daily_q",
    "merge_report",
    "compress_data",
    "remove_duplicates",
    "read_user_daily",
    "read_user_sample",
    "read_waterdata_daily",
    "read_waterdata_sample",
    "read_waterdata_info",
    "usgs_location_id",
    "populate_concentrations",
    "process_qw_data",
    "fix_sample_frame",
    "data_overview",
    "make_augmented_sample",
]

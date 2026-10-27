# PyEGRET tutorials

Each tutorial is a Jupyter notebook that runs from top to bottom on data
included in this folder. The notebooks are stored with their results, so you
can read them here on GitHub, figures and all, without running anything.

| | Tutorial | What it covers |
|---|---|---|
| 1 | [A WRTDS analysis from start to finish](01_choptank_walkthrough.ipynb) | The whole workflow on the Choptank River, following EGRET's own vignette: reading data, fitting the model, checking it, annual and flow-normalized results, contour surfaces, generalized flow normalization and `run_pairs`, WRTDS-Kalman. About a minute to run. |
| 2 | [Bringing in your own data](02_your_own_data.ipynb) | File formats, discharge units and date formats, censored ("less than") values, constituents reported in parts, converting a laboratory export, site information, checking the record before fitting, and the period of analysis. |

More are planned: downloading data from USGS, interpreting trends with
`run_pairs`, `run_groups` and `run_series`, WRTDS-Kalman load estimates, and
model diagnostics.

## Running them

**On your own computer.** With the repository downloaded, open a notebook from
this folder in Jupyter (`pip install jupyterlab`, then `jupyter lab`). The
first cell finds PyEGRET in the folder above, so nothing needs installing.

**In Google Colab.** Use the "Open in Colab" button at the top of each
notebook. The first cell then installs PyEGRET from GitHub and reads the data
from GitHub.

## The data

`data/choptank_daily.csv` and `data/choptank_sample.csv` are the daily
discharge (cubic feet per second) and the nitrate plus nitrite samples (mg/L
as N) for USGS site 01491000, Choptank River near Greensboro, Maryland,
water years 1980 to 2011. They are the `Choptank_eList` dataset distributed
with the R package EGRET, written out as plain CSV by
`diagnostics/export_choptank_csv.py`. They are U.S. Geological Survey data in
the public domain.

## For contributors

`python diagnostics/run_tutorials.py` runs every notebook and reports any that
fail; with `--save` it also stores the fresh results in the notebooks. Code in
the notebooks follows the same style rules as the package, and
`diagnostics/check_style.py` checks it.

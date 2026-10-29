# Installing PyEGRET

PyEGRET needs **Python 3.9 or later** and four common scientific packages:
numpy, pandas, scipy and matplotlib. Two more are optional:
[dataretrieval](https://github.com/DOI-USGS/dataretrieval-python), the USGS
package, if you want to download data from USGS, and `pytest` and `rdata` if
you want to run the tests.

There are two ways to use PyEGRET. Both give exactly the same code.

## 1. Install it with pip

```
pip install "pyegret @ git+https://github.com/gopalam/PyEGRET"
```

To include the USGS data readers, or the test tools:

```
pip install "pyegret[usgs] @ git+https://github.com/gopalam/PyEGRET"
pip install "pyegret[usgs,test] @ git+https://github.com/gopalam/PyEGRET"
```

A particular release can be installed by adding its tag, for example
`...PyEGRET@v0.2.0`. To update later, run the same command with `--upgrade`.

In a conda environment, install the four dependencies with conda first
(`conda install numpy pandas scipy matplotlib`) and then run the pip command.

## 2. Use it without installing

Download the repository (the green **Code** button on GitHub, then
**Download ZIP**, or `git clone`), unzip it anywhere, and put that folder on
Python's path at the top of your script or notebook:

```python
import sys
sys.path.insert(0, r"C:\path\to\PyEGRET")   # the folder that contains pyegret/

import pyegret
print(pyegret.__version__)
```

Install the dependencies yourself, for example with
`pip install -r requirements.txt` from inside the folder.

This is convenient when you want to read or change the code: edits take
effect the next time Python starts, with nothing to reinstall.

## Checking that it works

From the PyEGRET folder:

```
python -m pytest                      # the test suite, about four minutes, offline
python examples/choptank_wrtds.py     # the full workflow on the Choptank River
```

The example writes its figures to `examples/output/`.

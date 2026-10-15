"""
etunc.legacy
============
Unmaintained and untested code, kept so the exploratory notebooks and the
deprecated scripts in scripts/legacy_regrid/ can still import it. Nothing in
the binning pipeline (prep_*.py, make_edges.py, bin_all.py) uses it. One
module per origin:

- `notebook`: notebook-era loaders and helpers (the former `etunc/legacy.py`)
- `binning`, `grid`, `temporal`: functions replaced when the binning pipeline
  was restructured (docs/binning-workflow.md)
"""

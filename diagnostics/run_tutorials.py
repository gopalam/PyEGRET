"""
Run every tutorial notebook from top to bottom and report any that fail.

Use it after changing PyEGRET, to check that the tutorials still work, and
with ``--save`` to store the fresh results in the notebooks, so that the
figures and tables shown on GitHub match the current code.

Needs nbformat, nbclient and ipykernel (pip install nbclient ipykernel).

Notebooks that download data (marked ``"pyegret": {"needs_internet": true}``
in their metadata) can be left out with ``--offline``, or run alone with
``--internet``; name notebooks on the command line to run only those.

Run:  python diagnostics/run_tutorials.py             # check only
      python diagnostics/run_tutorials.py --save      # check and save outputs
      python diagnostics/run_tutorials.py --offline   # skip the downloads
      python diagnostics/run_tutorials.py 03_usgs_data.ipynb
Exit status is 0 when every notebook runs, 1 otherwise.
"""
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTORIALS = os.path.join(ROOT, "tutorials")


def run_notebook(path: str, save: bool) -> str:
    """Execute one notebook; return an empty string, or the error."""
    import nbclient
    import nbformat
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=1800, kernel_name="python3",
                                     resources={"metadata": {"path": TUTORIALS}})
    try:
        client.execute()
    except CellExecutionError as error:
        return str(error)
    if save:
        # Keep the saved notebooks free of run-specific metadata.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.metadata = {}
        nbformat.write(notebook, path)
    return ""


def needs_internet(path: str) -> bool:
    """True if the notebook's metadata says it downloads data."""
    import json

    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    settings = notebook["metadata"].get("pyegret", {})
    return bool(settings.get("needs_internet", False))


def main() -> int:
    save = "--save" in sys.argv
    offline = "--offline" in sys.argv
    internet_only = "--internet" in sys.argv
    names = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    paths = []
    for path in sorted(glob.glob(os.path.join(TUTORIALS, "*.ipynb"))):
        if names and os.path.basename(path) not in names:
            continue
        online = needs_internet(path)
        if offline and online:
            print(f"skipped {os.path.basename(path)} (needs the internet)")
            continue
        if internet_only and not online:
            continue
        paths.append(path)

    failed = 0
    for path in paths:
        start = time.time()
        error = run_notebook(path, save)
        name = os.path.basename(path)
        seconds = time.time() - start
        if error:
            failed = failed + 1
            print(f"FAILED  {name} ({seconds:.0f} s)")
            print(error)
        else:
            print(f"ok      {name} ({seconds:.0f} s)")
    print(f"{len(paths) - failed} of {len(paths)} notebooks ran")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Run every tutorial notebook from top to bottom and report any that fail.

Use it after changing PyEGRET, to check that the tutorials still work, and
with ``--save`` to store the fresh results in the notebooks, so that the
figures and tables shown on GitHub match the current code.

Needs nbformat, nbclient and ipykernel (pip install nbclient ipykernel).

Run:  python diagnostics/run_tutorials.py           # check only
      python diagnostics/run_tutorials.py --save    # check and save outputs
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


def main() -> int:
    save = "--save" in sys.argv
    paths = sorted(glob.glob(os.path.join(TUTORIALS, "*.ipynb")))
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

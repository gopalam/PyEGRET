"""
Check the code against the style rules in CLAUDE_CODE_INSTRUCTIONS.md.

Reports, for pyegret/, tests/, diagnostics/ and examples/:

- any class definition (no object-oriented code);
- any method called on the result of another call, e.g.
  ``df.sort_values("Date").reset_index()`` or ``np.isnan(x).sum()``
  (no chained calls);
- that pyegret imports from the project folder without being installed.

Run:  python diagnostics/check_style.py
Exit status is 0 when everything is clean, 1 otherwise.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ["pyegret", "tests", "diagnostics", "examples"]


def _called_on_a_call(node) -> bool:
    """True if this attribute is reached from the result of a call."""
    value = node
    while isinstance(value, (ast.Attribute, ast.Subscript)):
        value = value.value
        if isinstance(value, ast.Call):
            return True
    return False


def check_file(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [(error.lineno or 0, f"syntax error (Python {sys.version_info[0]}.{sys.version_info[1]}): {error.msg}")]
    lines = source.splitlines()
    problems = []
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            problems.append((node.lineno, f"class {node.name}"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if _called_on_a_call(node.func) and node.lineno not in seen:
                seen.add(node.lineno)
                problems.append((node.lineno, "chained call: " + lines[node.lineno - 1].strip()))
    return problems


def main() -> int:
    total = 0
    for folder in FOLDERS:
        for directory, _, files in os.walk(os.path.join(ROOT, folder)):
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(directory, name)
                for line, text in check_file(path):
                    print(f"{os.path.relpath(path, ROOT)}:{line}: {text}")
                    total = total + 1
    # PyEGRET must import straight from the folder, with nothing installed.
    # Run from a neutral folder with PYTHONPATH cleared, so that only the
    # sys.path.insert below can find it (but user-installed numpy etc. still load).
    import subprocess
    import tempfile
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import pyegret; "
             "assert pyegret.__file__.startswith(sys.argv[1])")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", probe, ROOT],
                            capture_output=True, text=True, env=env,
                            cwd=tempfile.gettempdir())
    if result.returncode != 0:
        print("pyegret does not import from the project folder:")
        print(result.stderr)
        total = total + 1
    print(f"{total} problem(s)")
    if total > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

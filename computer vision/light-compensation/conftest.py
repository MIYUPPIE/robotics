"""Makes `src`, `evals` and `tools` importable when pytest runs from the repo root
(the project directory name has a hyphen, so it cannot be a package itself)."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#!/usr/bin/env python
"""Plot logs/light_history.csv: level vs reference, gain, state bands, output error.

Usage: python tools/plot_history.py [history.csv] [-o out.png]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUMERIC = ("t_s", "level_filtered", "ref_level", "ratio", "gain", "brightness_error_pct")


def load_history(path: Path, session: str | None = None) -> dict[str, list]:
    """Columns as lists. Empty cells become NaN. Defaults to the most recent session."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    session = session or rows[-1]["session"]
    rows = [r for r in rows if r["session"] == session]
    out: dict[str, list] = {k: [] for k in (*NUMERIC, "state")}
    for r in rows:
        for k in NUMERIC:
            out[k].append(float(r[k]) if r[k] != "" else float("nan"))
        out["state"].append(r["state"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("history", nargs="?", default=str(ROOT / "logs" / "light_history.csv"))
    p.add_argument("-o", "--out", default=str(ROOT / "logs" / "history.png"))
    args = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_history(Path(args.history))
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(d["t_s"], d["level_filtered"], label="current level")
    ax[0].plot(d["t_s"], d["ref_level"], "--", label="reference")
    ax[0].set_ylabel("luminance (L*/255)")
    ax[0].legend()
    ax[1].plot(d["t_s"], d["gain"], color="tab:orange")
    ax[1].set_ylabel("compensation gain")
    ax[2].plot(d["t_s"], d["brightness_error_pct"], color="tab:red")
    ax[2].axhline(15, ls=":", color="gray")
    ax[2].set_ylabel("output error %")
    ax[2].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

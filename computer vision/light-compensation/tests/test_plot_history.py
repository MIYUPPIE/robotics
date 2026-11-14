import math

import pytest
from src.logger import HISTORY_COLUMNS
from tools.plot_history import load_history, main


def write(p, rows):
    header = ",".join(HISTORY_COLUMNS)
    p.write_text(header + "\n" + "\n".join(rows) + "\n")


ROW = "{s},x,{t},1,RUNNING,{st},0.5,0.5,0.6,0.8,1.2,1,1,0,0.6,{e},30,3"


def test_loads_latest_session_and_nan(tmp_path):
    p = tmp_path / "h.csv"
    write(
        p,
        [
            ROW.format(s="A", t=0, st="NORMAL", e=1),
            ROW.format(s="B", t=1, st="DARK", e=""),
            ROW.format(s="B", t=2, st="DARK", e=2),
        ],
    )
    d = load_history(p)
    assert d["t_s"] == [1.0, 2.0] and d["state"] == ["DARK", "DARK"]
    assert math.isnan(d["brightness_error_pct"][0])
    assert load_history(p, "A")["t_s"] == [0.0]


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "h.csv"
    write(p, [])
    with pytest.raises(ValueError):
        load_history(p)


def test_plot_writes_png(tmp_path):
    p = tmp_path / "h.csv"
    write(p, [ROW.format(s="A", t=i, st="NORMAL", e=1) for i in range(5)])
    out = tmp_path / "o.png"
    assert main([str(p), "-o", str(out)]) == 0 and out.stat().st_size > 1000

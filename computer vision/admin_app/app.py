"""Admin results dashboard Tkinter app. Run: `python -m admin_app.app`.

PIN-gated (ADMIN_PIN_HASH env var, see .env.example). Not voter-facing.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from admin_app.auth import PinGate
from admin_app.export import export_tallies_csv
from admin_app.tally import per_office_tally, turnout
from storage import db

DATA_DIR = Path(__file__).parent.parent / "data"
REFRESH_INTERVAL_MS = 2000


def _load_party_names() -> dict[str, str]:
    parties = json.loads((DATA_DIR / "parties_nigeria.json").read_text())
    return {p["acronym"]: p["name"] for p in parties}


def _load_office_names() -> dict[str, str]:
    offices = json.loads((DATA_DIR / "offices.json").read_text())
    return {o["code"]: o["name_en"] for o in offices}


class PinPrompt(tk.Toplevel):
    def __init__(self, master: tk.Tk, gate: PinGate, on_success):
        super().__init__(master)
        self.title("Admin PIN")
        self.gate = gate
        self.on_success = on_success
        self.geometry("300x140")
        self.protocol("WM_DELETE_WINDOW", master.destroy)

        ttk.Label(self, text="Enter admin PIN:").pack(pady=(20, 5))
        self.pin_var = tk.StringVar()
        entry = ttk.Entry(self, textvariable=self.pin_var, show="*")
        entry.pack(pady=5)
        entry.bind("<Return>", lambda _e: self._submit())
        entry.focus_set()
        ttk.Button(self, text="Unlock", command=self._submit).pack(pady=10)
        self.status = ttk.Label(self, text="", foreground="red")
        self.status.pack()

    def _submit(self) -> None:
        if self.gate.try_pin(self.pin_var.get()):
            self.destroy()
            self.on_success()
            return
        if self.gate.locked:
            self.status.configure(text="Too many attempts. Closing.")
            self.after(1500, self.master.destroy)
        else:
            remaining = self.gate.max_attempts - self.gate.attempts
            self.status.configure(text=f"Incorrect PIN ({remaining} attempt(s) left)")
        self.pin_var.set("")


class AdminApp(tk.Tk):
    def __init__(self, db_path: str | Path | None = None):
        super().__init__()
        self.title("Voter Kiosk Admin (Prototype)")
        self.geometry("900x600")
        self.withdraw()  # hidden until PIN is verified

        resolved_db_path = db_path or db.default_db_path()
        print(f"[admin_app] using database: {resolved_db_path}")
        self.conn = db.connect(resolved_db_path)
        self.party_names = _load_party_names()
        self.office_names = _load_office_names()

        stored_hash = os.environ.get("ADMIN_PIN_HASH", "")
        if not stored_hash:
            messagebox.showerror("Not configured", "ADMIN_PIN_HASH is not set -- see .env.example.")
            self.destroy()
            return

        gate = PinGate(stored_hash)
        PinPrompt(self, gate, self._unlock)

    def _unlock(self) -> None:
        self.deiconify()
        self._build_widgets()
        self._refresh()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=10)
        self.turnout_label = ttk.Label(top, text="", font=("TkDefaultFont", 12, "bold"))
        self.turnout_label.pack(side=tk.LEFT)
        ttk.Button(top, text="Export CSV", command=self._on_export).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.trees: dict[str, ttk.Treeview] = {}
        for code, name in self.office_names.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            tree = ttk.Treeview(frame, columns=("party", "votes"), show="headings")
            tree.heading("party", text="Party")
            tree.heading("votes", text="Votes")
            tree.pack(fill=tk.BOTH, expand=True)
            self.trees[code] = tree

    def _refresh(self) -> None:
        tally = per_office_tally(self.conn)
        for office_code, tree in self.trees.items():
            tree.delete(*tree.get_children())
            office_tally = tally.get(office_code, {})
            for party_code, count in sorted(office_tally.items(), key=lambda kv: -kv[1]):
                party_name = self.party_names.get(party_code, party_code)
                tree.insert("", tk.END, values=(f"{party_code} -- {party_name}", count))

        voted, registered, percent = turnout(self.conn)
        self.turnout_label.configure(
            text=f"Registered: {registered}   Voted: {voted}   Turnout: {percent:.1f}%"
        )
        self.after(REFRESH_INTERVAL_MS, self._refresh)

    def _on_export(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="tallies.csv")
        if not path:
            return
        export_tallies_csv(self.conn, path, self.party_names)
        messagebox.showinfo("Export complete", f"Tallies exported to {path}")


if __name__ == "__main__":  # pragma: no cover -- real-hardware entrypoint, not gate-tested
    AdminApp().mainloop()

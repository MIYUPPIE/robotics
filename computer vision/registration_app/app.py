"""Registration desk Tkinter app.

Officer-facing: keyboard/mouse form entry, with an embedded webcam preview
for face capture + liveness. Run: `python -m registration_app.app`.
"""

from __future__ import annotations

import json
import tkinter as tk
from datetime import date, datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from biometric.capture import burst_capture, capture_liveness
from biometric.embedding import FaceEmbedder
from biometric.face_mesh import FaceMeshTracker
from contracts.schemas import Gender, Language
from registration_app.capture_flow import enroll
from registration_app.form import RegistrationFormData, validate
from registration_app.slip import save_slip
from registration_app.webcam_widget import WebcamWidget
from storage import db

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models" / "mediapipe"
SLIPS_DIR = Path(__file__).parent / "slips"


def _load_states() -> list[dict]:
    return json.loads((DATA_DIR / "states_nigeria.json").read_text())


class RegistrationApp(tk.Tk):
    def __init__(self, db_path: str | Path | None = None):
        super().__init__()
        self.title("Voter Registration (Prototype -- not an official INEC system)")
        self.geometry("1150x720")

        resolved_db_path = db_path or db.default_db_path()
        print(f"[registration_app] using database: {resolved_db_path}")
        self.conn = db.connect(resolved_db_path)
        self.states = _load_states()
        self.state_codes = {s["code"] for s in self.states}

        # Heavy models are loaded lazily on first capture, not at startup,
        # so the form opens instantly.
        self._embedder: FaceEmbedder | None = None
        self._face_mesh: FaceMeshTracker | None = None

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _ensure_models_loaded(self) -> None:
        if self._embedder is None:
            self._embedder = FaceEmbedder()
        if self._face_mesh is None:
            self._face_mesh = FaceMeshTracker(MODELS_DIR / "face_landmarker.task")

    def _build_widgets(self) -> None:
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.webcam = WebcamWidget(right)
        self.webcam.pack(fill=tk.BOTH, expand=True)
        self.webcam.start()

        self.surname_var = tk.StringVar()
        self.first_name_var = tk.StringVar()
        self.middle_name_var = tk.StringVar()
        self.dob_var = tk.StringVar()
        self.gender_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.lga_var = tk.StringVar()
        self.phone_var = tk.StringVar()
        self.language_var = tk.StringVar(value=Language.ENGLISH.value)

        self.surname_entry = self._labeled_entry(left, "Surname*", self.surname_var)
        self._labeled_entry(left, "First name*", self.first_name_var)
        self._labeled_entry(left, "Middle name", self.middle_name_var)
        self._labeled_entry(left, "Date of birth (YYYY-MM-DD)*", self.dob_var)

        ttk.Label(left, text="Gender*").pack(anchor=tk.W)
        ttk.Combobox(
            left, textvariable=self.gender_var, values=[g.value for g in Gender], state="readonly"
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="State*").pack(anchor=tk.W)
        ttk.Combobox(
            left, textvariable=self.state_var, values=sorted(self.state_codes), state="readonly"
        ).pack(fill=tk.X, pady=(0, 8))

        self._labeled_entry(left, "LGA*", self.lga_var)
        self._labeled_entry(left, "Phone", self.phone_var)

        ttk.Label(left, text="Language*").pack(anchor=tk.W)
        ttk.Combobox(
            left,
            textvariable=self.language_var,
            values=[lang.value for lang in Language],
            state="readonly",
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Button(left, text="Capture Face + Register", command=self._on_register).pack(
            fill=tk.X, pady=(20, 0)
        )
        self.status_label = ttk.Label(left, text="", wraplength=320)
        self.status_label.pack(fill=tk.X, pady=(10, 0))

    def _labeled_entry(self, parent: tk.Widget, label: str, var: tk.StringVar) -> ttk.Entry:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        entry = ttk.Entry(parent, textvariable=var)
        entry.pack(fill=tk.X, pady=(0, 8))
        return entry

    def _current_form(self) -> RegistrationFormData:
        try:
            dob = date.fromisoformat(self.dob_var.get().strip())
        except ValueError:
            dob = None
        gender = Gender(self.gender_var.get()) if self.gender_var.get() else None
        language = Language(self.language_var.get()) if self.language_var.get() else None
        return RegistrationFormData(
            surname=self.surname_var.get(),
            first_name=self.first_name_var.get(),
            middle_name=self.middle_name_var.get(),
            dob=dob,
            gender=gender,
            state_code=self.state_var.get(),
            lga=self.lga_var.get(),
            phone=self.phone_var.get(),
            language_pref=language,
        )

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)
        self.update_idletasks()

    def _on_register(self) -> None:
        form = self._current_form()
        errors = validate(form, self.state_codes, date.today())
        if errors:
            messagebox.showerror(
                "Fix these fields", "\n".join(f"{e.field}: {e.message}" for e in errors)
            )
            return

        self._set_status("Loading face models (first capture only)...")
        self._ensure_models_loaded()
        assert self._embedder is not None and self._face_mesh is not None

        self._set_status("Please look at the camera and blink naturally...")
        liveness = capture_liveness(self.webcam.current_frame, self._face_mesh)
        if not liveness.is_live:
            messagebox.showwarning(
                "Liveness check failed",
                "We couldn't confirm a blink. Please try again, looking directly at the camera.",
            )
            self._set_status("")
            return

        self._set_status("Capturing face...")
        captured = burst_capture(self.webcam.current_frame, self._embedder)
        if not captured:
            messagebox.showwarning(
                "Capture failed", "No clear, single-face frame was captured. Try again."
            )
            self._set_status("")
            return

        outcome = enroll(self.conn, form, captured, liveness, now=datetime.now())

        if not outcome.success and outcome.duplicate_voter_id:
            self._handle_duplicate(form, captured, liveness, outcome)
            return

        if not outcome.success:
            messagebox.showerror("Registration failed", outcome.reason)
            self._set_status("")
            return

        self._finish_registration(outcome)

    def _handle_duplicate(self, form, captured, liveness, outcome) -> None:
        proceed = messagebox.askyesno(
            "Possible duplicate registration",
            f"This face closely matches an already-registered voter (ID: "
            f"{outcome.duplicate_voter_id}, similarity: {outcome.duplicate_similarity:.2f}).\n\n"
            "Reject this registration as a duplicate? Choose 'No' only if you, the "
            "registration officer, are certain this is a different person.",
            default="yes",
        )
        if proceed:
            self._set_status("Registration rejected as a likely duplicate.")
            return

        note = (
            simpledialog.askstring(
                "Override reason", "Enter a short reason for overriding the duplicate flag:"
            )
            or "(no reason given)"
        )
        override_outcome = enroll(
            self.conn,
            form,
            captured,
            liveness,
            now=datetime.now(),
            override_duplicate=True,
            officer_note=note,
        )
        if override_outcome.success:
            self._finish_registration(override_outcome)
        else:
            messagebox.showerror("Registration failed", override_outcome.reason)
            self._set_status("")

    def _finish_registration(self, outcome) -> None:
        assert outcome.voter is not None
        slip_path = save_slip(outcome.voter, SLIPS_DIR)
        messagebox.showinfo(
            "Registration complete",
            f"Voter ID: {outcome.voter.voter_id}\nSlip saved to: {slip_path}",
        )
        self._clear_form()
        self._set_status(f"Registered {outcome.voter.voter_id}. Ready for the next voter.")

    def _clear_form(self) -> None:
        """Resets every field so the previous voter's data can never bleed
        into the next registration by accident."""
        self.surname_var.set("")
        self.first_name_var.set("")
        self.middle_name_var.set("")
        self.dob_var.set("")
        self.gender_var.set("")
        self.state_var.set("")
        self.lga_var.set("")
        self.phone_var.set("")
        self.language_var.set(Language.ENGLISH.value)
        self.surname_entry.focus_set()

    def _on_close(self) -> None:
        self.webcam.stop()
        if self._face_mesh is not None:
            self._face_mesh.close()
        self.conn.close()
        self.destroy()


if __name__ == "__main__":  # pragma: no cover -- real-hardware entrypoint, not gate-tested
    RegistrationApp().mainloop()

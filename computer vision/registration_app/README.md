# registration_app

Tkinter registration desk app. Run: `python -m registration_app.app`.

- `form.py` -- form fields + a pure `validate()` function (required fields, 18+ age via
  `contracts.schemas.is_age_eligible`, valid state code, Nigerian phone format). No
  Tkinter dependency, so testable without any widget.
- `capture_flow.py` -- `enroll()`: the full registration decision path (liveness gate →
  duplicate check → voter ID generation → insert), composed from `biometric/*` and
  `storage/*`. Takes already-captured frames/liveness-result as input, so it's
  gate-tested with fakes.
- `webcam_widget.py` -- embedded OpenCV-in-Tkinter preview (standard `Label` +
  `.after()` pattern). Not gate-tested (needs a display/camera).
- `slip.py` -- PIL-rendered PNG registration slip, clearly marked
  "PROTOTYPE SYSTEM -- NOT AN OFFICIAL INEC DOCUMENT".
- `app.py` -- wires the form + webcam + `biometric.capture.capture_liveness` /
  `burst_capture` + `capture_flow.enroll` + `slip.save_slip` together. Duplicate
  matches are shown to the officer with Reject / Override(+reason, audited) options.

## Tests

`pytest registration_app/` -- `test_form_validation.py` (pure validation, every
required-field/age/state/phone case) and `test_capture_flow.py` (the full enroll
decision path against a real in-memory DB, with fake captured frames/liveness
results -- duplicate detection, override auditing, liveness gating). No Tkinter, no
camera, no real model, <2s.

# Eval fixture data -- consent requirements

This directory holds real recorded hand-tracking sessions used by
`gesture_dwell_eval.py`. It is gitignored (everything except `*.md` files) --
never commit real recordings here.

## Before adding anyone's recorded session

Get explicit, informed, written consent, stating clearly the recording will be
used to locally test gesture-selection accuracy for a prototype system and
won't be shared, published, or used for any other purpose. See
`biometric/evals/data/CONSENT.md` for the same policy applied to face data --
identical reasoning applies here (a hand-tracking recording plus its labeled
intent is still personal behavioral data).

## Expected layout once populated

`sessions/*.json`, each a labeled cursor-sample stream:

```json
{
  "regions": [
    { "region_id": "PARTY_APC", "x0": 100, "y0": 100, "x1": 300, "y1": 300 },
    { "region_id": "PARTY_PDP", "x0": 400, "y0": 100, "x1": 600, "y1": 300 }
  ],
  "samples": [
    { "x": 150.0, "y": 150.0, "t": 0.0, "present": true },
    { "x": 152.0, "y": 149.0, "t": 0.033, "present": true }
  ],
  "intended_region_id": "PARTY_APC"
}
```

Set `"intended_region_id": null` for a session where the voter deliberately
passed a finger across a region without meaning to select it (tests the
false-trigger rate, not just correct selection).

## No data yet?

`gesture_dwell_eval.py` runs a clearly-labeled **synthetic plumbing check**
instead when this directory has no real sessions -- that exercises the same
DwellSelector code over randomized procedural cursor paths, but is explicitly
not a measurement of real camera/MediaPipe tracking accuracy.

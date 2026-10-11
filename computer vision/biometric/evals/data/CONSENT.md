# Eval fixture data -- consent requirements

This directory holds real, non-synthetic biometric data used by
`face_match_eval.py` and `liveness_eval.py`. It is gitignored (everything
except `*.md` files) -- never commit real face images or recordings here.

## Before adding anyone's face/video to this directory

1. Get **explicit, informed, written consent** from every person, stating
   clearly that their image/video will be used to locally test face-matching
   and liveness-detection accuracy for a prototype system, and will not be
   shared, published, or used for any other purpose.
2. Record that consent (who, when, what they agreed to) -- e.g. a
   `consent_log.md` in this directory (also gitignored, since it will name
   real people).
3. Only use data you have the right to use -- don't scrape images from the
   internet or reuse a dataset without checking its license permits this use.
4. Delete the data when you're done evaluating, unless the person has agreed
   to longer-term retention.

## Expected layout once populated

**Face matching** (`face_match_eval.py`): `manifest.json` mapping an opaque
identity label to a list of image paths (relative to this directory), with at
least 2 images per identity so genuine (same-identity) pairs can be formed:

```json
{
  "identity_01": ["identity_01/photo1.jpg", "identity_01/photo2.jpg"],
  "identity_02": ["identity_02/photo1.jpg", "identity_02/photo2.jpg"]
}
```

**Liveness** (`liveness_eval.py`): `liveness_sessions/*.json`, each a recorded
landmark stream with a label:

```json
{ "should_blink": true, "landmarks": [[[x, y], ...478 points...], ...frames...] }
```

Include both real blink clips (`should_blink: true`) and static-photo-replay
attempts (`should_blink: false`) so the eval measures both blink recall and
spoof rejection.

## No data yet?

Both eval scripts run a clearly-labeled **synthetic plumbing check** instead
when this directory has no real data -- that verifies the eval math itself is
correct, but is explicitly not a measurement of real-world accuracy. Add real
consented data here (following the steps above) to get a real measurement.

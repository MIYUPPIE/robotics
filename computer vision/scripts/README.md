# scripts

One-off setup/ops scripts, not part of the concern-directory layout.

- `download_models.sh` -- fetches MediaPipe `.task` model files into `models/mediapipe/`
  (InsightFace's `buffalo_l` pack downloads itself on first use).
- `check_env.py` -- pre-flight check: camera, onnxruntime execution provider, model
  files, audio cache completeness, admin PIN config. Run before `make run-kiosk`.
- `seed_demo_data.py` -- seeds a few synthetic ("DemoVoterN") registered voters with
  fixture embeddings, so the full flow can be smoke-tested without real people.
  Refuses to run against a DB that already has real (non-demo) registrations unless
  `--force` is passed.

## Tests

`pytest scripts/` -- `seed_demo_data.py`'s pure demo-marker logic and its `seed()`
function against a real in-memory DB. `check_env.py` and `download_models.sh` are
real-environment inspection/setup tools, not gate-tested.

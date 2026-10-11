# Disclaimer

This repository is a **personal engineering prototype**. It demonstrates applying
computer vision (MediaPipe hand tracking), facial-recognition biometrics
(InsightFace), and Nigerian-language text-to-speech (YarnGPT) to the shape of a
Nigerian election ballot.

It is **not**:

- Official INEC (Independent National Electoral Commission) infrastructure.
- Affiliated with, endorsed by, or reviewed by INEC or any Nigerian government body.
- Security-audited, penetration-tested, or certified to any election-systems standard
  (there is no cryptographic end-to-end verifiability, no tamper-evident audit log
  beyond the local SQLite file, no multi-party ceremony for key management, no
  hardware security module -- none of the controls a real election system requires).
- Suitable for collecting real votes, real voter registrations, or any data from real
  people without their informed, explicit consent for a clearly-labeled demo/test.

It **is**:

- A local, single-machine, offline-capable kiosk demo.
- Built with real tests and real evals for the parts that are testable without a live
  camera/microphone, and clearly-labeled manual QA steps for the parts that aren't.
- Explicit, in `docs/NIGERIA_ELECTION_MODEL.md`, about every place it simplifies away
  from how a real Nigerian election actually works.

If you fork this for anything beyond a personal demo, re-read this file first, and
assume every simplification listed in `docs/NIGERIA_ELECTION_MODEL.md` needs real
engineering work before it's safe to rely on.

## Party logos

`data/party_logos/` contains real party logo images for the parties where a
verifiable, appropriately-licensed source was found (see
[../data/party_logos/ATTRIBUTION.md](../data/party_logos/ATTRIBUTION.md) for the
per-file source, license, and credit -- most are reproduced under Wikipedia's
non-free-logo fair-use rationale, one is CC BY-SA 4.0). This project reproduces
them at small UI-thumbnail resolution under a claimed **nominative/fair-use**
rationale: identifying real political parties by their own public mark, in a
personal, non-commercial, clearly-labeled kiosk demo. This is not a license
grant from any party, and the fair-use rationale does **not** travel with a
fork -- redistributing or deploying this beyond a personal demo requires
independently clearing rights for every logo used, or dropping back to the
color+acronym fallback that renders automatically for any party without a
sourced logo asset (see `hand_ui/renderer.py::draw_region`). Parties with no
logo asset are listed in `scripts/fetch_party_logos.py::NO_VERIFIED_LOGO` --
their logo was not fabricated or guessed at.

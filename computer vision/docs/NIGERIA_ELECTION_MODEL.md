# Nigeria election model -- what's real, what's simplified

This prototype follows the real shape of a Nigerian general election closely enough to
be a meaningful demo, but takes deliberate shortcuts documented here so nobody mistakes
it for authoritative.

## What's modeled faithfully

- **Five elected offices**, in the order a voter encounters them in this kiosk:
  President, Governor, Senator, House of Representatives member, State House of
  Assembly member -- matching INEC's real structure (`data/offices.json`).
- **Party-based ballot**: INEC ballots are dominated by party identity (logo + name),
  which is why this kiosk's ballot screens are party selection, not a written-in
  candidate choice.
- **18+ voting age**, enforced from date of birth at registration
  (`biometric/voter_id.py::is_age_eligible`).
- **36 states + FCT** as the state list (`data/states_nigeria.json`).
- **Real, currently/recently INEC-registered party names and acronyms**
  (`data/parties_nigeria.json`) -- reconfirm against INEC's current register before
  relying on this list for anything beyond a demo; party registration status changes
  (deregistrations, mergers) over time and this list is a point-in-time snapshot.

## What's deliberately simplified

- **Single combined session.** Real Nigerian elections run federal polls (President +
  National Assembly) and state polls (Governor + State Assembly) on separate days.
  This kiosk combines all 5 offices into one voting session for demo purposes.
- **One national party list reused across all 5 offices.** In reality, candidate
  rosters differ per state and per constituency, and not every party fields a
  candidate at every level. Modeling real per-state/per-constituency rosters would
  require a large, frequently-changing dataset that's out of scope for a prototype.
- **Party choice only, no named candidates.** This kiosk never displays or invents a
  candidate's name tied to a real office or constituency -- only the party. This is a
  deliberate choice to avoid fabricating claims about who is running for what.
- **LGA is free text**, not a validated Local Government Area / Ward / Polling Unit
  hierarchy. A real system would need the full INEC administrative hierarchy dataset.
- **Voter ID format is intentionally NOT INEC-VIN-shaped.** Real INEC Voter
  Identification Numbers are 19 digits with a defined internal structure. This
  prototype's ID (`PVR-{STATE}-{YYYYMMDD}-{6-char code}`, see `biometric/voter_id.py`)
  is deliberately different so it can never be mistaken for, or misused as, a real VIN.
- **No cryptographic ballot verifiability.** Ballot secrecy here means "the `votes`
  table has no voter reference" (see `docs/ARCHITECTURE.md#ballot-secrecy`), which is a
  reasonable software-engineering control for a demo but is not the same as the
  cryptographic guarantees (e.g. end-to-end verifiable voting, mixnets, zero-knowledge
  proofs) a real electronic voting system would need.
- **Single kiosk, single polling unit.** No multi-machine tallying, no network
  transmission of results, no chain-of-custody procedures.

If any of the above ever needs to change for a use beyond a personal demo, treat it as
a from-scratch requirements exercise, not an incremental patch.

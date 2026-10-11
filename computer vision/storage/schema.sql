-- Voter Kiosk CV database schema.
-- See docs/ARCHITECTURE.md#ballot-secrecy for why `votes` carries no voter
-- reference of any kind, and why `voted_log` is a separate table.

CREATE TABLE IF NOT EXISTS voters (
    voter_id                 TEXT PRIMARY KEY,
    surname                  TEXT NOT NULL,
    first_name                TEXT NOT NULL,
    middle_name                TEXT,
    dob                        TEXT NOT NULL,             -- ISO 'YYYY-MM-DD'
    gender                     TEXT NOT NULL CHECK (gender IN ('M', 'F', 'OTHER')),
    state_code                 TEXT NOT NULL,
    lga                        TEXT NOT NULL,
    phone                      TEXT,
    language_pref              TEXT NOT NULL CHECK (language_pref IN ('en', 'yo', 'ig', 'ha')),
    face_embedding              BLOB NOT NULL,             -- 512 x float32, np.ndarray.tobytes()
    embedding_model_version     TEXT NOT NULL,
    registered_at                TEXT NOT NULL,            -- ISO8601
    has_voted                    INTEGER NOT NULL DEFAULT 0 CHECK (has_voted IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_voters_state ON voters(state_code);

-- Anonymous vote records — NO voter reference of any kind. `cast_at` is rounded
-- to the nearest 15-minute bucket by the writer (storage/repositories/vote_repository.py)
-- as a residual mitigation against timing correlation with voted_log.cast_at.
CREATE TABLE IF NOT EXISTS votes (
    vote_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    office      TEXT NOT NULL CHECK (office IN ('PRESIDENT', 'GOVERNOR', 'SENATE', 'HOR', 'SHA')),
    party_code  TEXT NOT NULL,
    cast_at     TEXT NOT NULL                              -- ISO8601, 15-min bucketed
);

CREATE INDEX IF NOT EXISTS idx_votes_office_party ON votes(office, party_code);

-- Audit-only: blocks a second full voting session for the same voter. Contains
-- no vote content, so an exact timestamp here reveals nothing about a ballot.
CREATE TABLE IF NOT EXISTS voted_log (
    voter_id TEXT PRIMARY KEY REFERENCES voters(voter_id),
    cast_at  TEXT NOT NULL                                 -- ISO8601, exact
);

-- Registration-officer audit trail: every time a flagged duplicate-face match
-- is overridden (officer judges it's not actually a duplicate), logged here.
CREATE TABLE IF NOT EXISTS registration_overrides (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    new_voter_id      TEXT NOT NULL,
    matched_voter_id  TEXT NOT NULL,
    similarity        REAL NOT NULL,
    officer_note      TEXT,
    overridden_at     TEXT NOT NULL
);

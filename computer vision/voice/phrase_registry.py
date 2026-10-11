"""Phrase registry: loads data/phrases/{lang}.json.

Static phrases (canned sentences) and fragment phrases (`PARTY_NAME_*`,
`OFFICE_NAME_*`, composed at runtime into dynamic confirmations -- see
voice/playback.py::play_sequence) are distinguished only by naming
convention; both live in the same per-language JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contracts.schemas import Language

DATA_DIR = Path(__file__).parent.parent / "data"
PHRASES_DIR = DATA_DIR / "phrases"

# Default YarnGPT2 speaker per language. YarnGPT2's general-purpose speaker
# set (idera/emma/jude/osagie/tayo/zainab/joke/regina/remi/umar/chinenye) can
# render any of the 4 languages; these defaults just pick a
# culturally-associated voice per language. Confirm against
# https://yarngpt.ai/api-docs at integration time -- this was implemented
# against the publicly documented language/speaker support, not a live call.
DEFAULT_SPEAKERS: dict[Language, str] = {
    Language.ENGLISH: "emma",
    Language.YORUBA: "idera",
    Language.IGBO: "chinenye",
    Language.HAUSA: "zainab",
}


@dataclass(frozen=True)
class Phrase:
    phrase_id: str
    text: str
    review_status: str


def load_phrases(language: Language) -> dict[str, Phrase]:
    path = PHRASES_DIR / f"{language.value}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        phrase_id: Phrase(
            phrase_id=phrase_id, text=entry["text"], review_status=entry["review_status"]
        )
        for phrase_id, entry in raw.items()
    }


def canonical_phrase_ids() -> set[str]:
    """The full set of phrase_ids, sourced from the English (canonical) file."""
    return set(load_phrases(Language.ENGLISH).keys())


def is_fragment(phrase_id: str) -> bool:
    return phrase_id.startswith("PARTY_NAME_") or phrase_id.startswith("OFFICE_NAME_")


def party_acronyms() -> list[str]:
    parties = json.loads((DATA_DIR / "parties_nigeria.json").read_text(encoding="utf-8"))
    return [p["acronym"] for p in parties]


def office_codes() -> list[str]:
    offices = json.loads((DATA_DIR / "offices.json").read_text(encoding="utf-8"))
    return [o["code"] for o in offices]

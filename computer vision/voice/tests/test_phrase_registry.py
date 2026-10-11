from contracts.schemas import Language
from voice.phrase_registry import (
    DEFAULT_SPEAKERS,
    canonical_phrase_ids,
    is_fragment,
    load_phrases,
    office_codes,
    party_acronyms,
)


class TestPhraseCompleteness:
    def test_every_language_has_exactly_the_canonical_phrase_ids(self):
        canonical = canonical_phrase_ids()
        assert canonical, "canonical phrase set must not be empty"
        for language in Language:
            phrases = load_phrases(language)
            assert set(phrases.keys()) == canonical, (
                f"{language.value}.json phrase_id set differs from canonical: "
                f"missing={canonical - set(phrases.keys())}, "
                f"extra={set(phrases.keys()) - canonical}"
            )

    def test_every_phrase_has_nonempty_text(self):
        for language in Language:
            for phrase in load_phrases(language).values():
                assert phrase.text.strip(), f"{language.value}/{phrase.phrase_id} has empty text"

    def test_english_is_marked_source(self):
        for phrase in load_phrases(Language.ENGLISH).values():
            assert phrase.review_status == "source"

    def test_translations_are_marked_unreviewed(self):
        for language in (Language.YORUBA, Language.IGBO, Language.HAUSA):
            for phrase in load_phrases(language).values():
                assert phrase.review_status == "machine_draft_unreviewed"


class TestFragmentPhrases:
    def test_party_name_fragments_exist_for_every_party(self):
        canonical = canonical_phrase_ids()
        for acronym in party_acronyms():
            phrase_id = f"PARTY_NAME_{acronym}"
            assert phrase_id in canonical, f"missing fragment phrase {phrase_id}"
            assert is_fragment(phrase_id)

    def test_office_name_fragments_exist_for_every_office(self):
        canonical = canonical_phrase_ids()
        for code in office_codes():
            phrase_id = f"OFFICE_NAME_{code}"
            assert phrase_id in canonical, f"missing fragment phrase {phrase_id}"
            assert is_fragment(phrase_id)

    def test_static_phrases_are_not_flagged_as_fragments(self):
        assert not is_fragment("WELCOME_ATTRACT")
        assert not is_fragment("THANK_YOU")


def test_default_speaker_configured_for_every_language():
    assert set(DEFAULT_SPEAKERS.keys()) == set(Language)
    for speaker in DEFAULT_SPEAKERS.values():
        assert speaker

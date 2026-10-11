from datetime import date

import pytest

from contracts.schemas import Gender, Language
from registration_app.form import RegistrationFormData, validate

VALID_STATES = {"LAG", "OYO", "FCT"}
TODAY = date(2026, 8, 1)


def _form(**overrides) -> RegistrationFormData:
    defaults = dict(
        surname="Okafor",
        first_name="Ada",
        middle_name="",
        dob=date(2000, 1, 1),
        gender=Gender.FEMALE,
        state_code="LAG",
        lga="Ikeja",
        phone="",
        language_pref=Language.ENGLISH,
    )
    defaults.update(overrides)
    # mypy can't verify per-field types through a dict-unpack factory pattern.
    return RegistrationFormData(**defaults)  # type: ignore[arg-type]


class TestValidForm:
    def test_fully_valid_form_has_no_errors(self):
        assert validate(_form(), VALID_STATES, TODAY) == []

    def test_optional_fields_may_be_blank(self):
        errors = validate(_form(middle_name="", phone=""), VALID_STATES, TODAY)
        assert errors == []


class TestRequiredFields:
    @pytest.mark.parametrize(
        "field,value",
        [("surname", ""), ("first_name", ""), ("lga", "")],
    )
    def test_missing_required_text_field(self, field, value):
        errors = validate(_form(**{field: value}), VALID_STATES, TODAY)
        assert any(e.field == field for e in errors)

    def test_missing_dob(self):
        errors = validate(_form(dob=None), VALID_STATES, TODAY)
        assert any(e.field == "dob" for e in errors)

    def test_missing_gender(self):
        errors = validate(_form(gender=None), VALID_STATES, TODAY)
        assert any(e.field == "gender" for e in errors)

    def test_missing_state(self):
        errors = validate(_form(state_code=""), VALID_STATES, TODAY)
        assert any(e.field == "state_code" for e in errors)

    def test_missing_language(self):
        errors = validate(_form(language_pref=None), VALID_STATES, TODAY)
        assert any(e.field == "language_pref" for e in errors)


class TestAgeEligibility:
    def test_under_18_rejected(self):
        errors = validate(_form(dob=date(2015, 1, 1)), VALID_STATES, TODAY)
        assert any(e.field == "dob" for e in errors)

    def test_exactly_18_today_accepted(self):
        errors = validate(_form(dob=date(2008, 8, 1)), VALID_STATES, TODAY)
        assert not any(e.field == "dob" for e in errors)

    def test_future_dob_rejected(self):
        errors = validate(_form(dob=date(2027, 1, 1)), VALID_STATES, TODAY)
        assert any(e.field == "dob" for e in errors)


class TestStateCode:
    def test_unknown_state_code_rejected(self):
        errors = validate(_form(state_code="ZZZ"), VALID_STATES, TODAY)
        assert any(e.field == "state_code" for e in errors)


class TestPhone:
    @pytest.mark.parametrize("phone", ["08031234567", "+2348031234567", "07011122233"])
    def test_valid_nigerian_numbers_accepted(self, phone):
        errors = validate(_form(phone=phone), VALID_STATES, TODAY)
        assert not any(e.field == "phone" for e in errors)

    @pytest.mark.parametrize("phone", ["12345", "08012", "abcdefghijk", "+1 555 123 4567"])
    def test_invalid_numbers_rejected(self, phone):
        errors = validate(_form(phone=phone), VALID_STATES, TODAY)
        assert any(e.field == "phone" for e in errors)

"""Registration form fields + pure validation.

No Tkinter dependency -- `validate()` is testable without instantiating any
widget (see registration_app/tests/test_form_validation.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from contracts.schemas import Gender, Language, is_age_eligible

# Common Nigerian mobile format: 0[7-9]xxxxxxxxx or +234[7-9]xxxxxxxxx (11/13 digits).
PHONE_PATTERN = re.compile(r"^(\+234|0)[7-9]\d{9}$")


@dataclass
class RegistrationFormData:
    surname: str
    first_name: str
    middle_name: str
    dob: date | None
    gender: Gender | None
    state_code: str
    lga: str
    phone: str
    language_pref: Language | None


@dataclass(frozen=True)
class ValidationError:
    field: str
    message: str


def validate(
    form: RegistrationFormData, valid_state_codes: set[str], as_of: date
) -> list[ValidationError]:
    errors: list[ValidationError] = []

    if not form.surname.strip():
        errors.append(ValidationError("surname", "Surname is required"))
    if not form.first_name.strip():
        errors.append(ValidationError("first_name", "First name is required"))
    if not form.lga.strip():
        errors.append(ValidationError("lga", "LGA is required"))

    if form.dob is None:
        errors.append(ValidationError("dob", "Date of birth is required"))
    elif form.dob > as_of:
        errors.append(ValidationError("dob", "Date of birth cannot be in the future"))
    elif not is_age_eligible(form.dob, as_of):
        errors.append(ValidationError("dob", "Voter must be at least 18 years old"))

    if form.gender is None:
        errors.append(ValidationError("gender", "Gender is required"))

    if not form.state_code:
        errors.append(ValidationError("state_code", "State is required"))
    elif form.state_code not in valid_state_codes:
        errors.append(ValidationError("state_code", f"Unknown state code: {form.state_code}"))

    if form.phone.strip() and not PHONE_PATTERN.match(form.phone.strip()):
        errors.append(
            ValidationError(
                "phone", "Phone number doesn't look like a valid Nigerian mobile number"
            )
        )

    if form.language_pref is None:
        errors.append(ValidationError("language_pref", "Language preference is required"))

    return errors

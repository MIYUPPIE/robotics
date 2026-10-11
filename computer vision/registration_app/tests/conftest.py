from datetime import date

import numpy as np
import pytest

from contracts.schemas import EMBEDDING_DIM, Gender, Language
from registration_app.form import RegistrationFormData
from storage import db


@pytest.fixture
def conn():
    connection = db.connect_memory()
    yield connection
    connection.close()


@pytest.fixture
def valid_form() -> RegistrationFormData:
    return RegistrationFormData(
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


def make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)

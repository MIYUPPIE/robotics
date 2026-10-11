from pathlib import Path

import cv2
import numpy as np

from kiosk_app.party_logos import load_party_logos

REAL_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _write_png(path: Path, shape: tuple[int, ...]) -> None:
    image = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 3 and shape[2] == 4:
        image[:, :, 3] = 255
    cv2.imwrite(str(path), image)


class TestLoadPartyLogos:
    def test_loads_a_bgra_logo_by_acronym(self, tmp_path):
        (tmp_path / "party_logos").mkdir()
        _write_png(tmp_path / "party_logos" / "APC.png", (10, 20, 4))
        records = [{"acronym": "APC", "logo": "party_logos/APC.png"}]

        logos = load_party_logos(tmp_path, records)

        assert set(logos) == {"APC"}
        assert logos["APC"].shape == (10, 20, 4)

    def test_skips_parties_with_no_logo_field(self, tmp_path):
        records = [{"acronym": "SDP"}]
        assert load_party_logos(tmp_path, records) == {}

    def test_skips_parties_whose_logo_file_is_missing(self, tmp_path):
        records = [{"acronym": "APC", "logo": "party_logos/does_not_exist.png"}]
        assert load_party_logos(tmp_path, records) == {}

    def test_adds_an_opaque_alpha_channel_to_a_3_channel_image(self, tmp_path):
        (tmp_path / "party_logos").mkdir()
        _write_png(tmp_path / "party_logos" / "LP.png", (10, 20, 3))
        records = [{"acronym": "LP", "logo": "party_logos/LP.png"}]

        logos = load_party_logos(tmp_path, records)

        assert logos["LP"].shape == (10, 20, 4)
        assert (logos["LP"][:, :, 3] == 255).all()

    def test_every_sourced_logo_in_the_real_data_dir_loads(self):
        import json

        records = json.loads((REAL_DATA_DIR / "parties_nigeria.json").read_text())
        expected_acronyms = {p["acronym"] for p in records if p.get("logo")}
        assert expected_acronyms, "expected at least one party with a sourced logo"

        logos = load_party_logos(REAL_DATA_DIR, records)

        assert set(logos) == expected_acronyms
        for acronym, image in logos.items():
            assert image.ndim == 3 and image.shape[2] == 4, acronym
            assert image.shape[0] > 0 and image.shape[1] > 0, acronym

"""Fetches real party logo images into data/party_logos/ and (re)writes
data/party_logos/ATTRIBUTION.md from the same source-of-truth table below.

Party logos are trademarks of their respective parties. This project reproduces
them at small, UI-thumbnail resolution under a claimed nominative/fair-use
rationale -- identifying real political parties by their own public mark, in a
personal, non-commercial, clearly-labeled kiosk demo (see
docs/DISCLAIMER.md#party-logos). That rationale does NOT travel with a fork:
anyone reusing this beyond a personal demo must independently clear rights for
each logo, or drop back to the color+acronym fallback that
kiosk_app/party_logos.py / hand_ui/renderer.py already support for parties with
no logo asset.

Not every party in data/parties_nigeria.json has a source here -- INEC-registered
parties without an easily verifiable, appropriately-licensed logo image are
deliberately left out rather than guessed at. The ballot renderer falls back to
a color+acronym card for those (see hand_ui/renderer.py::draw_region).

Re-run with: python -m scripts.fetch_party_logos
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
LOGOS_DIR = DATA_DIR / "party_logos"
USER_AGENT = (
    "VoterKioskCV-Prototype/0.1 "
    "(personal, non-commercial engineering demo; contact: yemokanlawon@gmail.com)"
)


@dataclass(frozen=True)
class LogoSource:
    acronym: str
    party_name: str
    image_url: str
    source_page: str
    license_note: str
    credit: str


# Sourced 2026-08-02 via the Wikipedia/Wikimedia API (see
# scripts/tests/test_fetch_party_logos.py for the query approach). Six of the
# seven are English Wikipedia "non-free logo" files (fair-use rationale: identify
# the article subject); YPP's is CC BY-SA 4.0 on Wikimedia Commons. Every
# license_note below is what Wikipedia/Commons itself records for the file --
# not a claim this project independently verified with the party.
LOGO_SOURCES: tuple[LogoSource, ...] = (
    LogoSource(
        acronym="APC",
        party_name="All Progressives Congress",
        image_url="https://upload.wikimedia.org/wikipedia/en/2/23/All_Progressives_Congress_logo.png",
        source_page="https://en.wikipedia.org/wiki/File:All_Progressives_Congress_logo.png",
        license_note="Fair use (Wikipedia non-free logo, used to identify the party's own article)",
        credit="apc.com.ng",
    ),
    LogoSource(
        acronym="PDP",
        party_name="Peoples Democratic Party",
        image_url=(
            "https://upload.wikimedia.org/wikipedia/en/6/62/"
            "Logo_of_the_Peoples_Democratic_Party_%28Nigeria%29.png"
        ),
        source_page=(
            "https://en.wikipedia.org/wiki/File:Logo_of_the_Peoples_Democratic_Party_(Nigeria).png"
        ),
        license_note="Fair use (Wikipedia non-free logo, used to identify the party's own article)",
        credit="peoplesdemocraticparty.com.ng",
    ),
    LogoSource(
        acronym="LP",
        party_name="Labour Party",
        image_url="https://upload.wikimedia.org/wikipedia/en/5/5f/Labour_Party_%28Nigeria%29_logo.png",
        source_page="https://en.wikipedia.org/wiki/File:Labour_Party_(Nigeria)_logo.png",
        license_note="Fair use (Wikipedia non-free logo, used to identify the party's own article)",
        credit="labourparty.com.ng",
    ),
    LogoSource(
        acronym="NNPP",
        party_name="New Nigeria Peoples Party",
        image_url=(
            "https://upload.wikimedia.org/wikipedia/en/3/32/"
            "Logo_of_the_New_Nigeria_People%27s_Party.png"
        ),
        source_page=(
            "https://en.wikipedia.org/wiki/File:Logo_of_the_New_Nigeria_People%27s_Party.png"
        ),
        license_note="Fair use (Wikipedia non-free logo, used to identify the party's own article)",
        credit="inecnigeria.org (party profile page)",
    ),
    LogoSource(
        acronym="APGA",
        party_name="All Progressives Grand Alliance",
        image_url="https://upload.wikimedia.org/wikipedia/en/b/b3/APGA_Nigeria_Logo.png",
        source_page="https://en.wikipedia.org/wiki/File:APGA_Nigeria_Logo.png",
        license_note="Fair use (Wikipedia non-free logo, used to identify the party's own article)",
        credit="realnewsmagazine.net",
    ),
    LogoSource(
        acronym="ADC",
        party_name="African Democratic Congress",
        image_url="https://upload.wikimedia.org/wikipedia/en/2/24/African_Democratic_Congress_logo.png",
        source_page="https://en.wikipedia.org/wiki/File:African_Democratic_Congress_logo.png",
        license_note="Fair use (Wikipedia non-free logo, used to identify the party's own article)",
        credit="adc.org.ng",
    ),
    LogoSource(
        acronym="YPP",
        party_name="Young Progressives Party",
        image_url="https://upload.wikimedia.org/wikipedia/commons/9/95/Ypp-logo-design.png",
        source_page="https://commons.wikimedia.org/wiki/File:Ypp-logo-design.png",
        license_note="CC BY-SA 4.0",
        credit="ypp.ng",
    ),
)

# INEC-registered parties in data/parties_nigeria.json with no entry above --
# no freely-reusable or clearly-licensed logo image was found for them via the
# Wikipedia/Wikimedia/Wikidata APIs as of the date this was last run. Do not
# fabricate a logo for a real party: leave these on the color+acronym fallback
# until a real, sourced image is available.
NO_VERIFIED_LOGO: tuple[str, ...] = (
    "SDP",
    "ADP",
    "AA",
    "AAC",
    "APM",
    "BP",
    "NRM",
    "ZLP",
)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        return response.read()


def _normalize_and_save(raw_bytes: bytes, dest: Path) -> None:
    from io import BytesIO

    image = Image.open(BytesIO(raw_bytes)).convert("RGBA")
    image.save(dest, format="PNG", optimize=True)


def _write_attribution() -> None:
    lines = [
        "# Party logo attribution",
        "",
        "Auto-generated by `scripts/fetch_party_logos.py` -- do not hand-edit, re-run the",
        "script instead (it rewrites this file from `LOGO_SOURCES`).",
        "",
        "These are real party trademarks, reproduced at small UI-thumbnail resolution",
        "under a claimed nominative/fair-use rationale for a personal, non-commercial",
        "kiosk demo -- see [../../docs/DISCLAIMER.md#party-logos]"
        "(../../docs/DISCLAIMER.md#party-logos).",
        "Not an official grant of rights by any party. All marks are property of their",
        "respective owners.",
        "",
        "| Acronym | Party | File | License (per source) | Credit | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for src in LOGO_SOURCES:
        lines.append(
            f"| {src.acronym} | {src.party_name} | `{src.acronym}.png` | "
            f"{src.license_note} | {src.credit} | [{src.source_page}]({src.source_page}) |"
        )
    lines.append("")
    lines.append(
        "Parties in `data/parties_nigeria.json` with no row above (currently: "
        + ", ".join(NO_VERIFIED_LOGO)
        + ") have no verified logo asset -- the ballot renders a color+acronym card for "
        "them instead (see `hand_ui/renderer.py::draw_region`)."
    )
    lines.append("")
    (LOGOS_DIR / "ATTRIBUTION.md").write_text("\n".join(lines))


def fetch_all(force: bool = False) -> None:
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    for src in LOGO_SOURCES:
        dest = LOGOS_DIR / f"{src.acronym}.png"
        if dest.exists() and not force:
            print(f"already present: {dest}")
            continue
        print(f"fetching {src.acronym}: {src.image_url}")
        raw = _download(src.image_url)
        _normalize_and_save(raw, dest)
    _write_attribution()
    print(f"done -- {len(LOGO_SOURCES)} logos in {LOGOS_DIR}, ATTRIBUTION.md refreshed")


if __name__ == "__main__":
    fetch_all(force="--force" in sys.argv)

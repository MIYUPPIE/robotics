"""Printable registration slip -- a PIL-rendered PNG, not a real INEC
document (see docs/DISCLAIMER.md)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from contracts.schemas import VoterRecord

SLIP_SIZE = (800, 500)
DISCLAIMER_TEXT = "PROTOTYPE SYSTEM -- NOT AN OFFICIAL INEC DOCUMENT"


_Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def _load_fonts() -> tuple[_Font, _Font, _Font]:
    try:
        return (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 28),
            ImageFont.truetype("DejaVuSans.ttf", 20),
            ImageFont.truetype("DejaVuSans.ttf", 14),
        )
    except OSError:
        default = ImageFont.load_default()
        return default, default, default


def render_slip(voter: VoterRecord) -> Image.Image:
    image = Image.new("RGB", SLIP_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _load_fonts()

    draw.rectangle([(0, 0), (SLIP_SIZE[0] - 1, SLIP_SIZE[1] - 1)], outline=(0, 0, 0), width=3)
    draw.text((40, 30), "Voter Registration Slip", font=title_font, fill=(0, 0, 0))
    draw.text((40, 80), voter.voter_id, font=title_font, fill=(30, 90, 180))

    lines = [
        f"Name: {voter.full_name}",
        f"Date of birth: {voter.dob.isoformat()}",
        f"State / LGA: {voter.state_code} / {voter.lga}",
        f"Language: {voter.language_pref.value}",
        f"Registered: {voter.registered_at.date().isoformat()}",
    ]
    y = 150
    for line in lines:
        draw.text((40, y), line, font=body_font, fill=(0, 0, 0))
        y += 34

    draw.text((40, SLIP_SIZE[1] - 50), DISCLAIMER_TEXT, font=small_font, fill=(180, 0, 0))
    return image


def save_slip(voter: VoterRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{voter.voter_id}.png"
    render_slip(voter).save(path)
    return path

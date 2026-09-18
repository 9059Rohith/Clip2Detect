"""Render a submission poster from the measured Clip2Detect example and live UI screenshot."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "clip2detect-poster.png"
DATA = json.loads((ROOT / "site/src/data/demo.json").read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path if path.exists() else "DejaVuSans.ttf"), size)


def main() -> None:
    canvas = Image.new("RGB", (1600, 2200), "#0d1417")
    draw = ImageDraw.Draw(canvas)
    lime, muted, white = "#baff75", "#9db2a6", "#f4f9f2"
    for x in range(80, 1600, 120):
        draw.line((x, 0, x, 2200), fill="#1b2929", width=1)
    draw.rounded_rectangle((70, 70, 520, 135), radius=16, fill="#1b3229", outline="#4a784e", width=2)
    draw.text((98, 88), "CLIP2DETECT  /  PROJECT DEMO", font=font(27, True), fill=lime)
    draw.text((70, 190), "FROM VIDEO", font=font(106, True), fill=white)
    draw.text((70, 305), "TO DETECTOR.", font=font(106, True), fill=lime)
    draw.text((76, 452), "A reproducible object detection pipeline", font=font(43, True), fill=white)
    draw.text((76, 515), "Collect  /  Label  /  Augment  /  Train  /  Evaluate", font=font(28), fill=muted)

    values = [
        ("32", "source frames"),
        ("96", "exact object boxes"),
        ("128", "augmented images"),
        (f"{DATA['eval']['map50'] * 100:.1f}%", "mAP@50"),
    ]
    for index, (value, label) in enumerate(values):
        x = 70 + index * 370
        draw.rounded_rectangle((x, 615, x + 348, 790), radius=17, fill="#192528", outline="#405443", width=2)
        draw.text((x + 25, 638), value, font=font(66, True), fill=lime)
        draw.text((x + 26, 730), label.upper(), font=font(20, True), fill=muted)

    screenshot = Image.open(ROOT / "docs/screenshots/desktop-overview.png").convert("RGB")
    screenshot = screenshot.crop((0, 0, 1440, 1180))
    screenshot.thumbnail((1450, 1135), Image.Resampling.LANCZOS)
    draw.rounded_rectangle((64, 852, 1536, 2010), radius=22, fill="#1a292b", outline="#4e6956", width=3)
    canvas.paste(screenshot, (75, 865))
    draw.rectangle((70, 1990, 1530, 2015), fill="#0d1417")
    draw.text((75, 2055), "PYTHON  ·  OPENCV  ·  NUMPY  ·  PILLOW  ·  ULTRALYTICS YOLO  ·  PYTORCH", font=font(22, True), fill=lime)
    draw.text((75, 2105), "Original deterministic gameplay fixture  |  Actual trained weights and validation results", font=font(20), fill=muted)
    canvas.save(OUT, optimize=True)
    site_poster = ROOT / "site/public/poster.png"
    site_poster.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(site_poster, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()

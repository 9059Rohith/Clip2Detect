"""Command-line entry point for training on a local video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import Settings, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a labeled video into a YOLO detector")
    parser.add_argument("video", type=Path, help="Local video file")
    parser.add_argument("--classes", required=True, help="Comma-separated object classes")
    parser.add_argument("--manifest", type=Path, help="JSON annotations for sampled frames")
    parser.add_argument("--openai", action="store_true", help="Label sampled frames with GPT-5.6")
    parser.add_argument("--output", type=Path, default=Path("runs") / "custom")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in the selected output")
    args = parser.parse_args()
    classes = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    if not classes or bool(args.manifest) == args.openai:
        parser.error("Provide classes and exactly one of --manifest or --openai")
    result = run(Settings(
        video=args.video, output=args.output, classes=classes,
        manifest=args.manifest, use_openai=args.openai,
        fps=args.fps, epochs=args.epochs, image_size=args.image_size,
        batch=args.batch, model=args.model, overwrite=args.overwrite,
    ))
    print(json.dumps(result, indent=2))

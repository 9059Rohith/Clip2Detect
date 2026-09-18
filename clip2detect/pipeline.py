"""Build a YOLO detector from a local video and an annotation source.

The included Fruit Catch sample supplies exact boxes. Other videos can use a
matching annotation manifest or the OpenAI vision labeling option.
"""

from __future__ import annotations

import base64
import json
import os
import random
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class Settings:
    video: Path
    output: Path
    classes: tuple[str, ...]
    manifest: Path | None = None
    use_openai: bool = False
    fps: float = 4
    epochs: int = 6
    image_size: int = 320
    batch: int = 8
    train_fraction: float = 0.8
    seed: int = 42
    model: str = "yolov8n.pt"
    target_map50: float = 0.5
    overwrite: bool = False


def collect(video: Path, destination: Path, fps: float) -> list[Path]:
    if fps <= 0:
        raise ValueError("Sampling FPS must be positive")
    stream = cv2.VideoCapture(str(video))
    if not stream.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    source_fps = stream.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        stream.release()
        raise ValueError("Video does not report a valid frame rate")
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    frame_number = 0
    next_sample = 0.0
    try:
        while True:
            ok, pixels = stream.read()
            if not ok:
                break
            elapsed = frame_number / source_fps
            if elapsed + 1e-8 >= next_sample:
                path = destination / f"frame_{len(paths) + 1:06d}.jpg"
                if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not write {path}")
                paths.append(path)
                next_sample += 1 / fps
            frame_number += 1
    finally:
        stream.release()
    if len(paths) < 2:
        raise ValueError("At least two sampled frames are needed")
    return paths


def _openai_boxes(path: Path, classes: tuple[str, ...]) -> list[dict]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI labeling")
    with Image.open(path) as image:
        width, height = image.size
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    schema = {
        "type": "json_schema", "name": "detected_objects", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"objects": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string", "enum": list(classes)},
                    "x1": {"type": "number"}, "y1": {"type": "number"},
                    "x2": {"type": "number"}, "y2": {"type": "number"},
                },
                "required": ["class_name", "x1", "y1", "x2", "y2"],
                "additionalProperties": False,
            }}},
            "required": ["objects"], "additionalProperties": False,
        },
    }
    payload = {
        "model": os.environ.get("OPENAI_LABEL_MODEL", "gpt-5.6-luna"),
        "store": False,
        "max_output_tokens": 1600,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": f"Locate visible objects belonging to {list(classes)} in this {width} by {height} image. Return tight pixel boxes only; ignore text and decorations."},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
        ]}],
        "text": {"format": schema},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=70) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"OpenAI labeling failed with HTTP {error.code}") from error
    text = "".join(
        part.get("text", "")
        for item in result.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    if not text:
        raise RuntimeError("OpenAI returned no label data")
    boxes = []
    for object_ in json.loads(text)["objects"]:
        coordinates = [object_[axis] for axis in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in coordinates):
            continue
        x1, y1, x2, y2 = coordinates
        corners = [max(0, min(width, x1)), max(0, min(height, y1)),
                   max(0, min(width, x2)), max(0, min(height, y2))]
        if corners[2] > corners[0] and corners[3] > corners[1]:
            boxes.append({"class": object_["class_name"], "xyxy": corners})
    return boxes


def label(frames: list[Path], classes: tuple[str, ...], manifest: Path | None,
          use_openai: bool) -> dict:
    if bool(manifest) == use_openai:
        raise ValueError("Choose exactly one labeling source: manifest or OpenAI")
    records = None
    if manifest:
        source = json.loads(manifest.read_text(encoding="utf-8"))
        records = {record["frame"]: record["objects"] for record in source["frames"]}
        if len(records) != len(frames):
            raise ValueError("Manifest frame count differs from sampled video")
    report = []
    for number, path in enumerate(frames, 1):
        objects = records[number] if records is not None else _openai_boxes(path, classes)
        with Image.open(path) as image:
            width, height = image.size
        encoded = []
        for item in objects:
            category = item["class"]
            if category not in classes:
                raise ValueError(f"Unknown class: {category}")
            x1, y1, x2, y2 = item["xyxy"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid object box in frame {number}")
            encoded.append(f"{classes.index(category)} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        path.with_suffix(".txt").write_text("\n".join(encoded) + "\n", encoding="utf-8")
        report.append({"frame": number, "objects": objects})
    return {"frames": report, "classes": list(classes), "label_source": "exact-manifest" if manifest else "openai-api"}


def augment(frames: list[Path], destination: Path, seed: int) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for number, path in enumerate(frames):
        image = Image.open(path).convert("RGB")
        labels = path.with_suffix(".txt").read_text(encoding="utf-8").strip().splitlines()
        generator = np.random.default_rng(seed + number)
        variants = {
            "flip": ImageOps.mirror(image),
            "bright": ImageEnhance.Brightness(image).enhance(0.8 + 0.4 * generator.random()),
            "contrast": ImageEnhance.Contrast(image).enhance(0.8 + 0.4 * generator.random()),
            "noise": Image.fromarray(np.clip(np.asarray(image, dtype=np.int16) + generator.normal(0, 10, (image.height, image.width, 3)), 0, 255).astype(np.uint8)),
        }
        for kind, variant in variants.items():
            output = destination / f"{path.stem}_{kind}.jpg"
            variant.save(output, quality=95)
            if kind == "flip":
                transformed = []
                for line in labels:
                    category, x, y, w, h = line.split()
                    transformed.append(f"{category} {1-float(x):.6f} {y} {w} {h}")
            else:
                transformed = labels
            output.with_suffix(".txt").write_text("\n".join(transformed) + "\n", encoding="utf-8")
            outputs.append(output)
    return outputs


def make_dataset(frames: list[Path], variations: list[Path], destination: Path,
                 classes: tuple[str, ...], fraction: float, seed: int) -> tuple[Path, int, int]:
    if not 0 < fraction < 1:
        raise ValueError("Train fraction must be between zero and one")
    groups: dict[str, list[Path]] = {frame.stem: [frame] for frame in frames}
    for image in variations:
        source = image.stem.rsplit("_", 1)[0]
        groups[source].append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    split = max(1, min(len(keys) - 1, int(len(keys) * fraction)))
    train_groups = set(keys[:split])
    counts = {"train": 0, "val": 0}
    for source, images in groups.items():
        partition = "train" if source in train_groups else "val"
        for image in images:
            image_dir = destination / "images" / partition
            label_dir = destination / "labels" / partition
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(image.with_suffix(".txt"), label_dir / f"{image.stem}.txt")
            counts[partition] += 1
    config = destination.parent / "dataset.yaml"
    config.write_text(yaml.safe_dump({"path": str(destination.resolve()), "train": "images/train", "val": "images/val", "names": {index: name for index, name in enumerate(classes)}}), encoding="utf-8")
    return config, counts["train"], counts["val"]


def train_and_evaluate(config: Path, settings: Settings) -> dict:
    from ultralytics import YOLO

    network = YOLO(settings.model)
    training = network.train(
        data=str(config), epochs=settings.epochs, imgsz=settings.image_size,
        batch=settings.batch, seed=settings.seed, workers=0, plots=False,
        project=str(settings.output.resolve()), name="training", exist_ok=True,
    )
    best = Path(training.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError("Training produced no best.pt weights")
    weights = settings.output / "weights"
    weights.mkdir(exist_ok=True)
    shutil.copy2(best, weights / "best.pt")
    validation = YOLO(str(weights / "best.pt")).val(
        data=str(config), imgsz=settings.image_size, batch=settings.batch,
        plots=False, project=str(settings.output.resolve()), name="validation", exist_ok=True,
    )
    box = validation.box
    map50 = float(box.map50)
    report = {
        "map50": round(map50, 4), "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
        "target_accuracy": settings.target_map50,
        "meets_target": map50 >= settings.target_map50,
        "per_class": [
            {"class": settings.classes[int(index)], "ap50": round(float(ap), 4)}
            for index, ap in zip(box.ap_class_index, box.ap50)
        ],
    }
    (settings.output / "eval_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run(settings: Settings) -> dict:
    if not settings.video.is_file():
        raise FileNotFoundError(settings.video)
    output = settings.output.resolve()
    if settings.video.resolve().is_relative_to(output):
        raise ValueError("Source video must be outside the run output directory")
    generated = [output / name for name in (
        "input.mp4", "frames", "augmented", "dataset", "training", "validation",
        "weights", "labels.json", "dataset.yaml", "eval_results.json",
    )]
    existing = [path for path in generated if path.exists()]
    if existing and not settings.overwrite:
        raise FileExistsError(f"Run output already exists at {output}; choose a new directory or use --overwrite")
    if settings.overwrite:
        for path in existing:
            if path.resolve().parent != output:
                raise ValueError(f"Refusing to remove path outside run output: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    video_copy = settings.output / "input.mp4"
    if settings.video.resolve() != video_copy.resolve():
        shutil.copy2(settings.video, video_copy)
    frames = collect(video_copy, settings.output / "frames", settings.fps)
    labels = label(frames, settings.classes, settings.manifest, settings.use_openai)
    (settings.output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    variations = augment(frames, settings.output / "augmented", settings.seed)
    dataset_config, train_count, val_count = make_dataset(
        frames, variations, settings.output / "dataset", settings.classes,
        settings.train_fraction, settings.seed,
    )
    results = train_and_evaluate(dataset_config, settings)
    return {"frames": len(frames), "objects": sum(len(item["objects"]) for item in labels["frames"]),
            "augmentations": len(variations), "train": train_count, "validation": val_count,
            "evaluation": results}

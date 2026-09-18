"""Export the measured Fruit Catch run to the interactive website."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "fruit-catch-demo"
SAMPLE = ROOT / "site" / "public" / "sample"


def main() -> None:
    labels = json.loads((RUN / "labels.json").read_text(encoding="utf-8"))
    evaluation = json.loads((RUN / "eval_results.json").read_text(encoding="utf-8"))
    frames = sorted((RUN / "frames").glob("frame_*.jpg"))
    variations = list((RUN / "augmented").glob("*.jpg"))
    train = list((RUN / "dataset" / "images" / "train").glob("*.jpg"))
    validation = list((RUN / "dataset" / "images" / "val").glob("*.jpg"))
    data = {
        "project": "fruit-catch-demo",
        "source": "Original generated Fruit Catch gameplay fixture",
        "frames": labels["frames"],
        "objectCount": sum(len(frame["objects"]) for frame in labels["frames"]),
        "augmentedCount": len(variations),
        "trainCount": len(train),
        "validationCount": len(validation),
        "eval": evaluation,
    }
    (ROOT / "site" / "src" / "data" / "demo.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (SAMPLE / "raw").mkdir(parents=True, exist_ok=True)
    (SAMPLE / "annotated").mkdir(parents=True, exist_ok=True)
    for path, record in zip(frames, labels["frames"], strict=True):
        shutil.copy2(path, SAMPLE / "raw" / path.name)
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for object_ in record["objects"]:
            draw.rectangle(object_["xyxy"], outline="#baff75" if object_["class"] == "fruit" else "#ffad6a", width=2)
        image.save(SAMPLE / "annotated" / path.name, quality=95)
    for source, name in (
        (RUN / "input.mp4", "input.mp4"),
        (RUN / "labels.json", "ground_truth.json"),
        (RUN / "eval_results.json", "eval_results.json"),
        (RUN / "weights" / "best.pt", "best.pt"),
    ):
        shutil.copy2(source, SAMPLE / name)
    print(f"Published {len(frames)} frames, {data['objectCount']} boxes, and measured mAP@50 {evaluation['map50']:.4f}")


if __name__ == "__main__":
    main()

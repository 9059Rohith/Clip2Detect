"""Check the two data boundaries that affect detector quality."""

from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from clip2detect.pipeline import _openai_boxes, label, make_dataset

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_openai_labels_use_image_input_and_validate_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.jpg"
            Image.new("RGB", (640, 360)).save(image)
            output = {"output": [{"content": [{"type": "output_text", "text": json.dumps({"objects": [
                {"class_name": "fruit", "x1": 10, "y1": 20, "x2": 70, "y2": 80},
                {"class_name": "bomb", "x1": -10, "y1": 30, "x2": 40, "y2": 90},
            ]})}]}]}

            def fake_request(request, timeout):
                self.assertEqual(timeout, 70)
                body = json.loads(request.data)
                self.assertEqual(body["model"], "gpt-5.6-luna")
                self.assertTrue(body["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
                return io.BytesIO(json.dumps(output).encode("utf-8"))

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_LABEL_MODEL": "gpt-5.6-luna"}):
                with mock.patch("urllib.request.urlopen", side_effect=fake_request):
                    boxes = _openai_boxes(image, ("fruit", "bomb"))
            self.assertEqual(boxes[0]["xyxy"], [10, 20, 70, 80])
            self.assertEqual(boxes[1]["xyxy"], [0, 30, 40, 90])

    def test_sample_annotations_encode_valid_yolo_boxes(self) -> None:
        truth = json.loads((ROOT / "examples/fruit-catch/ground_truth.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            for record in truth["frames"]:
                path = Path(directory) / f"frame_{record['frame']:06d}.jpg"
                Image.new("RGB", (640, 360)).save(path)
                frames.append(path)
            result = label(frames, ("fruit", "bomb"), ROOT / "examples/fruit-catch/ground_truth.json", False)
            self.assertEqual(len(result["frames"]), 32)
            self.assertEqual(sum(len(item["objects"]) for item in result["frames"]), 96)
            for path in frames:
                for row in path.with_suffix(".txt").read_text().splitlines():
                    category, x, y, width, height = map(float, row.split())
                    self.assertIn(category, (0.0, 1.0))
                    self.assertTrue(0 < x < 1 and 0 < y < 1)
                    self.assertTrue(0 < width < 1 and 0 < height < 1)

    def test_augmented_images_remain_with_source_in_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            frames, variants = [], []
            for number in range(1, 11):
                name = f"frame_{number:06d}"
                for suffix, target in (("", frames), ("_flip", variants), ("_noise", variants)):
                    image = temp / f"{name}{suffix}.jpg"
                    image.write_bytes(b"image")
                    image.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1\n")
                    target.append(image)
            _, train, val = make_dataset(frames, variants, temp / "dataset", ("fruit",), 0.8, 42)
            self.assertEqual((train, val), (24, 6))
            train_sources = {path.stem.rsplit("_", 1)[0] if path.stem.endswith(("_flip", "_noise")) else path.stem for path in (temp / "dataset/images/train").glob("*.jpg")}
            val_sources = {path.stem.rsplit("_", 1)[0] if path.stem.endswith(("_flip", "_noise")) else path.stem for path in (temp / "dataset/images/val").glob("*.jpg")}
            self.assertFalse(train_sources & val_sources)


if __name__ == "__main__":
    unittest.main()

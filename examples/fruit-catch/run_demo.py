"""Generate and train on Clip2Detect's original Fruit Catch fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clip2detect.pipeline import Settings, run  # noqa: E402
from create_fixture import main as create_fixture  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    create_fixture()
    report = run(Settings(
        video=here / "input.mp4",
        output=ROOT / "runs" / "fruit-catch-demo",
        classes=("fruit", "bomb"),
        manifest=here / "ground_truth.json",
        overwrite=True,
    ))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

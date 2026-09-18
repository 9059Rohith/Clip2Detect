"""Record the running Clip2Detect viewer, narrate it, and burn synchronized subtitles."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from mutagen.mp3 import MP3
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMP = HERE / "work"
TEMP.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SEGMENTS = [
    ("Clip2Detect turns video into an object detector. This real run includes source frames, exact labels, trained weights, and measured results.", "intro"),
    ("The pipeline has five stages: collect, label, augment, train, and evaluate. Each stage is available in the Python project and explained in this live artifact viewer.", "collect"),
    ("For this submission, we made an original eight second Fruit Catch gameplay clip. The collector sampled it at four frames per second, producing thirty two source frames.", "frames"),
    ("The frame browser lets us inspect each sampled image. Here is frame twelve, with fruit and bomb positions visibly different from the previous frame.", "frame"),
    ("Ground truth boxes come from the fixture's drawing coordinates and are encoded in YOLO format. These are verified, exact labels.", "label"),
    ("We can hide and restore those boxes to compare labels with the source pixels. The inspector reports two fruit and one bomb in this frame.", "toggle"),
    ("Here is a separate live OpenAI vision request. GPT five point six Luna analyzes the selected frame through a server side API, and its dashed boxes can be compared with the exact ground truth.", "ai_label"),
    ("The augmentation stage creates flipped, brighter, contrast adjusted, and noise varied images. It transforms each corresponding annotation along with the image, producing one hundred twenty eight variations.", "augment"),
    ("Next, Ultralytics and PyTorch train YOLO version eight nano for six epochs. The training set contains one hundred twenty five images, and validation contains thirty five.", "train"),
    ("We split by source frame group, keeping all variants of a source in the same partition. This avoids near duplicate leakage between training and validation.", "split"),
    ("The separate evaluation reports ninety nine point zero percent m A P at fifty, eighty seven point nine percent m A P across IoU thresholds, and ninety one point five percent precision.", "metrics"),
    ("Recall is ninety three point five percent on this small held out fixture. These are measured results from the trained model, but this controlled example is not a production accuracy benchmark.", "metrics2"),
    ("The export panel provides the real input video, ground truth JSON, evaluation JSON, and trained PyTorch weights. Each file downloads from the running site.", "export"),
    ("The summary ties the output back to the workflow: thirty two frames, ninety six exact boxes, one hundred twenty eight augmented images, and the trained detector.", "summary"),
    ("Codex assisted with the fixture, React viewer, Vercel API, tests, and recording. OpenCV, NumPy, PyTorch, and Ultralytics YOLO power the training pipeline.", "technology"),
    ("To reproduce this complete example locally, synchronize dependencies with uv, then run the included Fruit Catch script. It generates footage, labels, model weights, and evaluation from scratch.", "reproduce"),
    ("This demo shows the working site and its actual outputs. Explore the code and run the pipeline on your own video.", "ending"),
]


def srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


async def create_voice() -> list[float]:
    durations = []
    for index, (line, _) in enumerate(SEGMENTS):
        target = voice_path(index, line)
        if not target.exists():
            await edge_tts.Communicate(line, "en-US-AriaNeural", rate="+20%").save(str(target))
        durations.append(MP3(target).info.length)
        print(f"Voice {index + 1}/{len(SEGMENTS)}: {durations[-1]:.1f}s", flush=True)
    return durations


def voice_path(index: int, line: str) -> Path:
    digest = hashlib.sha256(line.encode("utf-8")).hexdigest()[:10]
    return TEMP / f"voice_{index:02}_{digest}.mp3"


def record(durations: list[float]) -> tuple[Path, list[float], float]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(TEMP), record_video_size={"width": 1280, "height": 720}, device_scale_factor=1, accept_downloads=True)
        startup_start = time.monotonic()
        page = context.new_page()
        page.goto(os.environ.get("CLIP2DETECT_DEMO_URL", "https://clip2detect.vercel.app/"), wait_until="networkidle")
        page.get_by_text("Server-side API ready").wait_for(timeout=15000)
        startup = time.monotonic() - startup_start
        scene_durations = []
        for index, (_, action) in enumerate(SEGMENTS):
            scene_start = time.monotonic()
            if action == "collect":
                page.locator(".pipeline-step").nth(0).click()
                page.locator(".pipeline").scroll_into_view_if_needed()
            elif action == "frames":
                page.locator(".workbench").scroll_into_view_if_needed()
            elif action == "frame":
                page.get_by_role("button", name="Show frame 12").click()
            elif action == "label":
                page.locator(".pipeline-step").nth(1).click()
                page.get_by_role("tab", name="annotations").click()
            elif action == "toggle":
                page.get_by_role("button", name="Hide labels").click()
                page.wait_for_timeout(1800)
                page.get_by_role("button", name="Show labels").click()
            elif action == "ai_label":
                page.get_by_role("button", name="Show frame 1", exact=True).click()
                page.get_by_role("button", name="Analyze this frame").click()
                page.locator(".annotation-box.is-ai").first.wait_for(timeout=60000)
                page.get_by_role("button", name="Show ground truth").click()
                page.wait_for_timeout(1200)
                page.get_by_role("button", name="Show AI boxes").click()
            elif action == "augment":
                page.locator(".pipeline-step").nth(2).click()
            elif action == "train":
                page.locator(".pipeline-step").nth(3).click()
            elif action == "split":
                page.locator(".phase-detail").scroll_into_view_if_needed()
            elif action == "metrics":
                page.locator(".pipeline-step").nth(4).click()
                page.get_by_role("tab", name="metrics").click()
                page.locator(".workbench").scroll_into_view_if_needed()
            elif action == "export":
                page.get_by_role("tab", name="export").click()
            elif action == "summary":
                page.locator(".summary-grid").scroll_into_view_if_needed()
            elif action == "technology":
                page.locator(".tech-strip").scroll_into_view_if_needed()
            elif action == "reproduce":
                page.locator(".run-panel").scroll_into_view_if_needed()
            elif action == "ending":
                page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            remaining = durations[index] - (time.monotonic() - scene_start)
            if remaining > 0:
                page.wait_for_timeout(int(remaining * 1000))
            scene_durations.append(time.monotonic() - scene_start)
            print(f"Recorded scene {index + 1}/{len(SEGMENTS)}", flush=True)
        raw_path = Path(page.video.path())
        context.close()
        browser.close()
    return raw_path, scene_durations, startup


def finish(raw: Path, voice_durations: list[float], scene_durations: list[float], startup: float) -> None:
    srt = HERE / "clip2detect-demo.srt"
    cursor = 0.0
    blocks = []
    subtitle_index = 1
    for (line, _), voice_duration, scene_duration in zip(SEGMENTS, voice_durations, scene_durations):
        words = line.split()
        groups = [words[index:index + 10] for index in range(0, len(words), 10)]
        start = cursor
        for group in groups:
            segment_duration = voice_duration * len(group) / len(words)
            end = start + segment_duration
            blocks.append(f"{subtitle_index}\n{srt_time(start)} --> {srt_time(end)}\n{' '.join(group)}\n")
            subtitle_index += 1
            start = end
        cursor += scene_duration
    srt.write_text("\n".join(blocks), encoding="utf-8")
    if cursor >= 180:
        raise RuntimeError(f"Demo is {cursor:.1f}s, above the three minute limit")
    concat = TEMP / "audio-list.txt"
    audio_parts = []
    for index, duration in enumerate(scene_durations):
        part = TEMP / f"voice_padded_{index:02}.wav"
        subprocess.run([FFMPEG, "-y", "-i", str(voice_path(index, SEGMENTS[index][0])), "-af", "apad", "-t", f"{duration:.3f}", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(part)], check=True, capture_output=True)
        audio_parts.append(part)
    concat.write_text("\n".join(f"file '{part.as_posix()}'" for part in audio_parts) + "\n", encoding="utf-8")
    narration = TEMP / "narration.wav"
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(narration)], check=True, capture_output=True)
    subtitle_path = str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
    out = HERE / "clip2detect-demo.mp4"
    filter_graph = f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=12,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=28'"
    subprocess.run([FFMPEG, "-y", "-ss", f"{startup:.3f}", "-i", str(raw), "-i", str(narration), "-vf", filter_graph, "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)], check=True, capture_output=True)
    print(f"Video: {out} ({cursor:.1f}s)")


def main() -> None:
    durations = asyncio.run(create_voice())
    print(f"Total narration: {sum(durations):.1f}s", flush=True)
    raw, scene_durations, startup = record(durations)
    finish(raw, durations, scene_durations, startup)


if __name__ == "__main__":
    main()

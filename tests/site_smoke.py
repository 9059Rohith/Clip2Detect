"""Browser smoke test and submission screenshots for the artifact viewer."""

from __future__ import annotations

import os
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
BASE_URL = os.environ.get("CLIP2DETECT_SITE_URL", "http://localhost:4173").rstrip("/")
DEMO = json.loads((ROOT / "site" / "src" / "data" / "demo.json").read_text(encoding="utf-8"))


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        errors: list[str] = []
        desktop.on("pageerror", lambda error: errors.append(str(error)))
        desktop.goto(BASE_URL + "/", wait_until="networkidle")
        assert desktop.get_by_text("From video to detector").is_visible()
        assert desktop.get_by_text(f"{DEMO['eval']['map50'] * 100:.1f}%", exact=False).count() >= 1
        assert desktop.locator(".frame-row").count() == 32
        desktop.screenshot(path=str(SHOTS / "desktop-overview.png"), full_page=True)
        desktop.get_by_role("button", name="Show frame 12").click()
        assert "000012" in desktop.locator(".viewer-meta").inner_text()
        desktop.get_by_role("button", name="Hide labels").click()
        assert desktop.locator(".annotation-box").count() == 0
        desktop.get_by_role("button", name="Show labels").click()
        assert desktop.locator(".annotation-box").count() == 3
        desktop.get_by_role("tab", name="metrics").click()
        assert desktop.get_by_text("Measured model performance").is_visible()
        desktop.screenshot(path=str(SHOTS / "evaluation.png"), full_page=False)
        desktop.get_by_role("tab", name="export").click()
        for name in ("Input video", "Ground truth", "Evaluation JSON", "Trained weights"):
            assert desktop.get_by_role("link", name=name).is_visible()
        desktop.get_by_role("button", name="Augment", exact=False).click()
        assert desktop.locator(".phase-detail h2").inner_text() == "Augment"
        desktop.get_by_role("button", name="Play frames").click()
        assert desktop.get_by_role("button", name="Pause frames").is_visible()
        desktop.get_by_role("button", name="Pause frames").click()
        for asset in ("/sample/input.mp4", "/sample/ground_truth.json", "/sample/eval_results.json", "/sample/best.pt"):
            response = desktop.request.get(BASE_URL + asset)
            assert response.status == 200, (asset, response.status)
        assert not errors, errors

        ai_page = browser.new_page(viewport={"width": 1440, "height": 900})
        def ai_route(route):
            if route.request.method == "GET":
                route.fulfill(json={"configured": True, "model": "gpt-5.6-luna"})
            else:
                route.fulfill(json={"frame": 1, "model": "gpt-5.6-luna", "source": "test-response", "objects": [{"class": "fruit", "xyxy": [78, 86, 130, 150]}]})
        ai_page.route("**/api/label", ai_route)
        ai_page.goto(BASE_URL + "/", wait_until="networkidle")
        ai_page.get_by_role("button", name="Analyze this frame").click()
        ai_page.locator(".annotation-box.is-ai").first.wait_for(timeout=15000)
        assert ai_page.locator(".annotation-box.is-ai").count() == 1
        ai_page.get_by_role("button", name="Show ground truth").click()
        assert ai_page.locator(".annotation-box.is-ai").count() == 0

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.goto(BASE_URL + "/", wait_until="networkidle")
        assert mobile.locator("body").evaluate("el => el.scrollWidth <= window.innerWidth")
        mobile.screenshot(path=str(SHOTS / "mobile-overview.png"), full_page=True)
        mobile.get_by_role("tab", name="metrics").click()
        assert mobile.get_by_text("Measured model performance").is_visible()
        browser.close()
    print("Browser smoke passed: desktop, mobile, controls, artifacts, and no page errors")


if __name__ == "__main__":
    main()

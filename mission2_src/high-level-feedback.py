#!/usr/bin/env python3
"""Classify mouth state on live frames via Anthropic Claude."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import time
from typing import Iterable, Optional

import cv2
import requests

FRAMES_PATH = os.environ.get("FASTVLM_FRAMES_PATH", "frames/current.jpg")
INTERVAL_SEC = float(os.environ.get("FASTVLM_INTERVAL_SEC", "20"))
SAMPLES_DIR = os.environ.get("FASTVLM_SAMPLES_DIR", "samples")
SAMPLES_BASENAME = os.environ.get("FASTVLM_SAMPLES_BASENAME", "latest.jpg")
VIDEO_SOURCE = os.environ.get("FASTVLM_VIDEO_SOURCE", "/dev/video5")
WARMUP_FRAMES = int(os.environ.get("FASTVLM_WARMUP_FRAMES", "5"))

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise SystemExit("ANTHROPIC_API_KEY manquant dans l'environnement.")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")

PROMPT = (
    "Réponds UNIQUEMENT avec un de ces deux tokens.\n"
    "Bouche fermée: YELLOW-MIRROR\n"
    "Bouche ouverte: BLUE-TWEEZER"
)

VALID_TOKENS = {"YELLOW-MIRROR", "BLUE-TWEEZER"}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 60
POLL_SLEEP = 1.0


def file_hash(path: str) -> str:
    """Return a deterministic hash for the given file path."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_text(content: Iterable[dict]) -> str:
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def call_claude(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        return "INVALID:FORMAT"

    with open(image_path, "rb") as handle:
        img_b64 = base64.b64encode(handle.read()).decode("ascii")

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": img_b64,
                        },
                    },
                ],
            }
        ],
    }

    try:
        response = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        error_body = exc.response.text if getattr(exc, "response", None) else str(exc)
        print(f"[Anthropic error] {error_body}")
        return f"ERROR:{exc}"

    text = _extract_text(response.json().get("content", []))
    if text in VALID_TOKENS:
        return text
    return f"INVALID:{text or 'EMPTY'}"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _init_video_capture() -> Optional[cv2.VideoCapture]:
    if not VIDEO_SOURCE:
        return None
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source vidéo: {VIDEO_SOURCE}")
    for _ in range(max(0, WARMUP_FRAMES)):
        cap.read()
    print(f"[Video] Source {VIDEO_SOURCE} initialisée.")
    return cap


def _capture_frame(cap: cv2.VideoCapture) -> bool:
    ret, frame = cap.read()
    if not ret:
        return False
    if FRAMES_PATH:
        _ensure_parent_dir(FRAMES_PATH)
        cv2.imwrite(FRAMES_PATH, frame)
    if SAMPLES_DIR:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        sample_path = os.path.join(SAMPLES_DIR, SAMPLES_BASENAME)
        cv2.imwrite(sample_path, frame)
    return True


def main() -> None:
    print("Live classifier started (Ctrl+C to stop)")
    last_hash: str | None = None
    cap = _init_video_capture()
    last_capture_ts = 0.0

    try:
        while True:
            now = time.time()
            if cap is not None:
                if now - last_capture_ts < INTERVAL_SEC:
                    time.sleep(POLL_SLEEP)
                    continue
                if not _capture_frame(cap):
                    time.sleep(POLL_SLEEP)
                    continue
                last_capture_ts = now
            elif not os.path.exists(FRAMES_PATH):
                time.sleep(POLL_SLEEP)
                continue

            try:
                current_hash = file_hash(FRAMES_PATH)
            except OSError:
                time.sleep(POLL_SLEEP)
                continue

            if cap is None and current_hash == last_hash:
                time.sleep(POLL_SLEEP)
                continue

            last_hash = current_hash
            result = call_claude(FRAMES_PATH)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp}\t{result}", flush=True)

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()

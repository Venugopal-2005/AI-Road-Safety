"""
Setup script: Copies BDD100K images to the real_images demo folder and
sets up a Pexels video symlink (or copy) to real_videos/.

Run from the project root:
    python scripts/setup_real_data.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BDD100K_SRC = REPO_ROOT / "assets" / "sample_images" / "archive" / "bdd100k" / "bdd100k" / "images" / "100k" / "train" / "trainA"
BDD100K_DST = REPO_ROOT / "assets" / "real_images" / "bdd100k"
REAL_VIDEOS_DIR = REPO_ROOT / "assets" / "real_videos"
SAMPLE_VIDEO_SRC = REPO_ROOT / "assets" / "sample_videos" / "sample_road_driving.mp4"
REAL_VIDEO_DST = REAL_VIDEOS_DIR / "urban_intersection.mp4"

NUM_IMAGES = 50


def copy_bdd100k_samples() -> None:
    BDD100K_DST.mkdir(parents=True, exist_ok=True)

    if not BDD100K_SRC.is_dir():
        print(f"[WARNING] BDD100K source not found: {BDD100K_SRC}")
        return

    images = list(BDD100K_SRC.glob("*.jpg"))
    if not images:
        print(f"[WARNING] No .jpg images found in {BDD100K_SRC}")
        return

    random.seed(42)
    selected = random.sample(images, min(NUM_IMAGES, len(images)))
    copied = 0
    for img in selected:
        dest = BDD100K_DST / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
            copied += 1

    print(f"[OK] Copied {copied} BDD100K images to {BDD100K_DST} (total: {len(list(BDD100K_DST.glob('*.jpg')))})")


def setup_real_video() -> None:
    REAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if REAL_VIDEO_DST.exists():
        print(f"[OK] Real video already exists: {REAL_VIDEO_DST}")
        return

    # If user placed Pexels video at real_videos/urban_intersection.mp4, nothing to do.
    # Otherwise fall back to copying sample video as placeholder.
    if SAMPLE_VIDEO_SRC.is_file():
        shutil.copy2(SAMPLE_VIDEO_SRC, REAL_VIDEO_DST)
        print(f"[INFO] Copied sample_road_driving.mp4 as placeholder for urban_intersection.mp4")
        print(f"       Replace {REAL_VIDEO_DST} with the Pexels traffic video for best results.")
    else:
        print(f"[WARNING] No source video found. Place your video at:\n  {REAL_VIDEO_DST}")


if __name__ == "__main__":
    print("=" * 60)
    print("AI Road Safety – Real Dataset Setup")
    print("=" * 60)
    copy_bdd100k_samples()
    setup_real_video()
    print("=" * 60)
    print("Setup complete.")

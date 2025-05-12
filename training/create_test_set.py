#!/usr/bin/env python3
"""
Move 10 % of images from
    processed_dataset/{hands,non_hands}/
into
    test_set/{hands,non_hands}/
Keeps directory structure; reproducible with a fixed RNG seed.
"""
import shutil, random, math
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────
SOURCE_DIR = Path(r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\processed_dataset")
TEST_DIR   = SOURCE_DIR.parent / "test_set"           # sibling folder
TEST_RATIO = 0.10
RNG_SEED   = 42
# ────────────────────────────────────────────────────────────

random.seed(RNG_SEED)
TEST_DIR.mkdir(parents=True, exist_ok=True)

moved_total = 0
for cls_dir in SOURCE_DIR.iterdir():                       # hands, non_hands
    if not cls_dir.is_dir(): 
        continue
    files = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
    n_test = math.ceil(len(files) * TEST_RATIO)
    test_files = random.sample(files, n_test)

    # make destination subfolder
    dest_cls_dir = TEST_DIR / cls_dir.name
    dest_cls_dir.mkdir(parents=True, exist_ok=True)

    for f in test_files:
        shutil.move(str(f), dest_cls_dir / f.name)
    moved_total += n_test
    print(f"Moved {n_test:4d} files → {dest_cls_dir}")

print(f"\nDone. Total moved to test set: {moved_total}")
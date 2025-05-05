#!/usr/bin/env python3
import os
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 1) EDIT THESE PATHS
# ─────────────────────────────────────────────────────────────
SRC_ROOT = r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\dataset"
DST_ROOT = r"C:\Users\User\OneDrive\Documents\IOT_ML\proj1-khayes39\proj1-kurt\training\processed_dataset"

CLASSES = ["hands", "non_hands"]
IMG_SIZE = (96, 96)

# ─────────────────────────────────────────────────────────────
# 2) PROCESSING LOOP
# ─────────────────────────────────────────────────────────────
for cls in CLASSES:
    src_dir = os.path.join(SRC_ROOT, cls)
    dst_dir = os.path.join(DST_ROOT, cls)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        src_path = os.path.join(src_dir, fname)
        # load → grayscale → resize
        img = Image.open(src_path).convert("L")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        # save as PNG (you can swap to .pgm if you prefer)
        base, _ = os.path.splitext(fname)
        dst_path = os.path.join(dst_dir, base + ".png")
        img.save(dst_path)

    print(f"[+] Processed {cls}: {len(os.listdir(dst_dir))} images → {dst_dir}")

print("✅ All classes processed.")
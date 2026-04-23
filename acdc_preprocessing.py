#!/usr/bin/env python3
"""
ACDC Cardiac MRI — Preprocessing Pipeline

Preprocesses the ACDC training dataset into 2D .npy slices ready for training.

What this script does:
- Creates a fixed 80/20 patient-level train/val split (saved to JSON)
- Clips intensity outliers per volume (1st-99th percentile)
- Z-score normalises each volume
- Resamples in-plane to 1.37 mm/pixel (dataset median spacing)
- Saves each slice as a .npy file under database/prepared_2d_v2/

Output structure:
    database/prepared_2d_v2/
    ├── train/
    │   ├── images/   # .npy files, one per slice
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── patient_split.json

Before running:
    Set ACDC_ROOT to your local ACDC directory.
"""

from __future__ import annotations
import json, random
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

# ── CONFIGURATION ────────────────────────────────────────────────────────────
ACDC_ROOT  = Path("/your/path/to/ACDC")  # <-- set this to your local path
RAW_SPLIT  = "training"

OUT_2D          = ACDC_ROOT / "database" / "prepared_2d_v2"
SPLIT_JSON      = OUT_2D / "patient_split.json"
RAW_TRAIN_DIR   = ACDC_ROOT / "database" / RAW_SPLIT

VAL_RATIO         = 0.2
SEED              = 42
TARGET_SPACING_XY = (1.37, 1.37)
CLIP_LOW          = 1
CLIP_HIGH         = 99

# ── PREPROCESSING FUNCTIONS ──────────────────────────────────────────────────
def normalize_volume_v2(vol: np.ndarray,
                         clip_low: int = 1,
                         clip_high: int = 99) -> np.ndarray:
    """
    Per-volume normalization: clip outliers then z-score.
    More robust than per-slice min-max which is sensitive to outlier pixels.
    """
    vol = vol.astype(np.float32)
    vol = np.clip(vol, np.percentile(vol, clip_low), np.percentile(vol, clip_high))
    std = vol.std()
    return ((vol - vol.mean()) / std).astype(np.float32) if std > 1e-8 \
           else np.zeros_like(vol)


def resample_volume_xy(img: np.ndarray,
                        msk: np.ndarray,
                        spacing: tuple,
                        target: tuple = (1.37, 1.37)):
    """Resample x/y axes to a common spacing. Z axis is left unchanged."""
    zx = spacing[0] / target[0]
    zy = spacing[1] / target[1]
    return (zoom(img, (zx, zy, 1.0), order=1).astype(np.float32),
            zoom(msk, (zx, zy, 1.0), order=0).astype(np.uint8))


def create_split(raw_dir: Path,
                  out_json: Path,
                  val_ratio: float = 0.2,
                  seed: int = 42) -> dict:
    """Create an 80/20 patient-level split and save to JSON. Loads if it exists."""
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if out_json.exists():
        split = json.load(open(out_json))
        print(f"Loaded existing split: train={len(split['train'])} | "
              f"val={len(split['val'])} patients")
        return split

    patients = sorted([p.name for p in Path(raw_dir).iterdir()
                        if p.is_dir() and p.name.startswith("patient")])
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_ratio)))
    split = {"train": patients[n_val:], "val": patients[:n_val], "seed": seed}
    json.dump(split, open(out_json, "w"), indent=2)
    print(f"Split saved: train={len(split['train'])} | "
          f"val={len(split['val'])} patients")
    return split


def export_patient(patient_name: str, split_name: str,
                    raw_train_dir: Path, out_2d: Path) -> int:
    """Preprocess and export all slices for one patient."""
    pdir    = raw_train_dir / patient_name
    img_out = out_2d / split_name / "images"
    msk_out = out_2d / split_name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(pdir.glob(f"{pdir.name}_frame*.nii.gz"))
    img_files = [f for f in nii_files if not f.name.endswith("_gt.nii.gz")]
    written = 0

    for img_path in img_files:
        msk_path = img_path.with_name(
            img_path.name.replace(".nii.gz", "_gt.nii.gz"))
        if not msk_path.exists():
            continue

        img_nii = nib.load(str(img_path))
        img_vol = img_nii.get_fdata().astype(np.float32)
        msk_vol = nib.load(str(msk_path)).get_fdata().astype(np.uint8)
        spacing = img_nii.header.get_zooms()[:3]

        img_vol, msk_vol = resample_volume_xy(
            img_vol, msk_vol, spacing, TARGET_SPACING_XY)
        img_vol = normalize_volume_v2(img_vol, CLIP_LOW, CLIP_HIGH)

        for k in range(img_vol.shape[-1]):
            name = img_path.name.replace(".nii.gz", f"_slice{k:03d}.npy")
            np.save(img_out / name, img_vol[..., k])
            np.save(msk_out / name, msk_vol[..., k])
            written += 1

    return written


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RAW_TRAIN_DIR.exists():
        raise FileNotFoundError(f"ACDC training data not found: {RAW_TRAIN_DIR}")

    # Check if already exported
    n_existing = len(list((OUT_2D / "train" / "images").glob("*.npy")))
    if n_existing > 0:
        print(f"Slices already exported ({n_existing} train slices found) — skipping.")
    else:
        split = create_split(RAW_TRAIN_DIR, SPLIT_JSON, VAL_RATIO, SEED)

        print("\nExporting training slices...")
        total_tr = sum(
            export_patient(name, "train", RAW_TRAIN_DIR, OUT_2D)
            for name in tqdm(split["train"], desc="train"))

        print("Exporting validation slices...")
        total_vl = sum(
            export_patient(name, "val", RAW_TRAIN_DIR, OUT_2D)
            for name in tqdm(split["val"], desc="val"))

        print(f"\nDone — train: {total_tr} slices | val: {total_vl} slices")
        print(f"Saved to: {OUT_2D}")

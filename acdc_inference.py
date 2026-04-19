#!/usr/bin/env python3
"""
ACDC Cardiac MRI Segmentation — Inference Script (Test Set: patient101–150)

This script applies the trained two-stage cascade to unseen test volumes and
saves per-frame segmentation predictions as NIfTI files.

Per-frame inference pipeline
-----------------------------
1. Preprocessing: identical to training (1st/99th percentile clip, global
   z-score normalization, in-plane resampling to TARGET_SPACING_XY).
2. Center pad/crop to 256×256 (TARGET_SIZE).
3. Stage-1 ensemble (5-fold): each fold produces a binary heart mask;
   a union bounding box across all folds is computed for robust localization.
4. Crop the 256×256 image to the bounding box and resize to 192×192 (CROP_SIZE).
5. Stage-2 ensemble (5-fold) with TTA (horizontal flip, vertical flip, both):
   4 augmented views × 5 models = 20 forward passes; probabilities are averaged.
6. Argmax over the averaged softmax to produce a 192×192 class label map.
7. Back-project the 192×192 prediction to the 256×256 canvas using exact
   offset inversion (no interpolation to avoid spatial misalignment).
8. Invert the pad/crop step to recover the resampled-space resolution.
9. Zoom back to original patient spacing.
10. 3D connected component analysis: retain only the largest connected
    component per class to remove spurious isolated predictions.

Parameters that MUST match the training configuration:
  CROP_MARGIN, TARGET_SPACING_XY, TARGET_SIZE, CROP_SIZE, S1_BBOX_THRESHOLD

Configuration:
  EXP_DIR  — path to the training run directory (contains fold_*/checkpoints/)
  OUT_DIR  — output directory for predicted NIfTI files
"""

from __future__ import annotations
import warnings, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.ndimage import zoom, label as nd_label
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import monai
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIGURATION ───────────────────────────────────────────────────────────
# Update EXP_DIR to point to the training run before executing.
ACDC_ROOT = Path("/Users/fabian/Desktop/ACDC")
EXP_DIR   = ACDC_ROOT / "runs" / "last_run_20260331_115106"  # set to the training run directory
OUT_DIR   = EXP_DIR / "predictions_test"

# These values must be identical to training — any mismatch causes silently wrong spatial alignment
TARGET_SPACING_XY = (1.37, 1.37)
TARGET_SIZE       = (256, 256)
CROP_SIZE         = (192, 192)
CROP_MARGIN       = 40
S1_BBOX_THRESHOLD = 0.4
K_FOLDS           = 5
USE_INSTANCE_NORM = False

# TTA: 4 augmentation variants (original, h-flip, v-flip, both); probabilities are averaged after inverse-flipping
TTA_FLIPS = [(False, False), (True, False), (False, True), (True, True)]

# ── DEVICE & AMP ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

USE_AMP = device.type == "cuda"
print(f"Device: {device} | AMP: {USE_AMP}")

# ── MODEL DEFINITIONS ───────────────────────────────────────────────────────
# Architecture must be identical to training — weights are loaded directly.
def _norm(ch: int) -> nn.Module:
    return nn.InstanceNorm2d(ch, affine=True) if USE_INSTANCE_NORM else nn.BatchNorm2d(ch)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), _norm(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), _norm(inter_ch))
        self.psi    = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), _norm(1), nn.Sigmoid())
        self.relu   = nn.ReLU(True)

    def forward(self, gate, skip):
        g = self.W_gate(gate); s = self.W_skip(skip)
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.psi(self.relu(g + s))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.attn = AttentionGate(in_ch // 2, skip_ch, skip_ch // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(True))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attn(gate=x, skip=skip)], 1))


class ResNetAttentionUNet_DS(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, dropout=0.1):
        super().__init__()
        r = tv_models.resnet34(weights=None)
        self.enc0 = nn.Sequential(r.conv1, r.bn1, r.relu); self.pool = r.maxpool
        self.enc1 = r.layer1; self.enc2 = r.layer2; self.enc3 = r.layer3; self.enc4 = r.layer4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True), nn.Dropout2d(dropout))
        self.dec4 = DecoderBlock(512, 256, 256); self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64); self.dec1 = DecoderBlock( 64,  64,  32)
        self.final_up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False), _norm(32), nn.ReLU(True),
            nn.Dropout2d(dropout), nn.Conv2d(32, num_classes, 1))
        self.ds_head4 = nn.Conv2d(256, num_classes, 1)
        self.ds_head3 = nn.Conv2d(128, num_classes, 1)
        self.ds_head2 = nn.Conv2d( 64, num_classes, 1)

    def forward(self, x):
        sz = x.shape[-2:]
        e0 = self.enc0(x); e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b, e3); d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1); d1 = self.dec1(d2, e0)
        out = self.final_up(d1)
        if out.shape[-2:] != sz:
            out = F.interpolate(out, size=sz, mode="bilinear", align_corners=False)
        return self.final_conv(out)

# ── CHECKPOINT LOADING ──────────────────────────────────────────────────────
def load_s1_models(exp_dir: Path, k: int) -> List[nn.Module]:
    """Load all k Stage-1 fold checkpoints and return them in eval mode.

    Each fold was trained independently; all k models are used as an ensemble
    at inference to reduce localization variance across unseen patients.
    """
    models = []
    for fi in range(k):
        ckpt_path = exp_dir / f"fold_{fi}" / "checkpoints" / f"best_s1_f{fi}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
        m = monai.networks.nets.UNet(
            spatial_dims=2, in_channels=3, out_channels=2,
            channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval().to(device)
        models.append(m)
    print(f"Stage-1: {len(models)} fold models loaded")
    return models


def load_s2_models(exp_dir: Path, k: int) -> List[nn.Module]:
    """Load all k Stage-2 fold checkpoints and return them in eval mode.

    pretrained=False at instantiation because weights are fully loaded from the
    checkpoint; ImageNet initialization is not needed here.
    """
    models = []
    for fi in range(k):
        ckpt_path = exp_dir / f"fold_{fi}" / "checkpoints" / f"best_s2_f{fi}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")
        m = ResNetAttentionUNet_DS(4, False, 0.1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval().to(device)
        models.append(m)
    print(f"Stage-2: {len(models)} fold models loaded")
    return models

# ── PREPROCESSING ───────────────────────────────────────────────────────────
def preprocess_volume(img_vol: np.ndarray, orig_spacing: Tuple) -> Tuple[np.ndarray, Tuple]:
    """Apply the same preprocessing pipeline used during training.

    Steps: 1st/99th percentile clip, global z-score normalization, in-plane
    resampling to TARGET_SPACING_XY. Returns the resampled volume and its
    spatial dimensions (H_res, W_res), which are required later to invert
    the pad/crop step during back-projection.
    """
    lo, hi = np.percentile(img_vol, 1.0), np.percentile(img_vol, 99.0)
    img_vol = np.clip(img_vol, lo, hi)
    mu, sig = img_vol.mean(), img_vol.std()
    img_vol = ((img_vol - mu) / sig).astype(np.float32) if sig > 1e-8 \
              else np.zeros_like(img_vol)

    sx, sy = orig_spacing[:2]
    zf = (sx / TARGET_SPACING_XY[0], sy / TARGET_SPACING_XY[1], 1.0)
    img_res = zoom(img_vol, zf, order=1).astype(np.float32)
    H_res, W_res = img_res.shape[:2]
    return img_res, (H_res, W_res)

# ── SPATIAL GEOMETRY HELPERS ────────────────────────────────────────────────
def pad_crop_2d(arr: np.ndarray, ts: Tuple[int, int]) -> np.ndarray:
    """Center-pad then center-crop to target size ts=(H, W).

    Padding is applied first so that inputs smaller than ts are handled correctly.
    Works for both (H, W) and (H, W, C) arrays.
    """
    th, tw = ts; h, w = arr.shape[:2]
    ph = max(0, th - h); pw = max(0, tw - w)
    if arr.ndim == 2:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    else:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    h, w = arr.shape[:2]
    return arr[(h - th) // 2:(h - th) // 2 + th, (w - tw) // 2:(w - tw) // 2 + tw]


def invert_pad_crop_2d(pred: np.ndarray, H_orig: int, W_orig: int) -> np.ndarray:
    """Exact inverse of pad_crop_2d: maps a 256×256 prediction back to H_orig×W_orig.

    Uses integer offset arithmetic only — no zoom or interpolation — to avoid
    sub-pixel spatial shifts that would misalign the prediction with the input geometry.
    """
    H_pad = max(TARGET_SIZE[0], H_orig)
    W_pad = max(TARGET_SIZE[1], W_orig)

    # recover the crop offset that was applied when going from padded to TARGET_SIZE
    crop_h = (H_pad - TARGET_SIZE[0]) // 2
    crop_w = (W_pad - TARGET_SIZE[1]) // 2

    # place the 256×256 prediction back into the padded canvas at the original crop position
    padded = np.zeros((H_pad, W_pad), dtype=pred.dtype)
    padded[crop_h:crop_h + TARGET_SIZE[0], crop_w:crop_w + TARGET_SIZE[1]] = pred

    # recover the padding offset that was added when going from H_orig to TARGET_SIZE
    pad_h = max(0, TARGET_SIZE[0] - H_orig)
    pad_w = max(0, TARGET_SIZE[1] - W_orig)
    start_h = pad_h // 2
    start_w = pad_w // 2

    return padded[start_h:start_h + H_orig, start_w:start_w + W_orig]


def compute_heart_bbox_union(
        preds_256: List[np.ndarray],  # one binary mask per fold, shape (256, 256)
        margin: int = CROP_MARGIN,
        h_max: int = TARGET_SIZE[0],
        w_max: int = TARGET_SIZE[1]) -> Tuple[int, int, int, int]:
    """Compute a union bounding box across all fold predictions.

    Taking the extremal coordinates over all 5 folds produces a conservative
    (larger) bounding box that is robust to uncertain slices where one or more
    folds may miss the heart. Falls back to a fixed 160×160 central crop when
    no cardiac structure is detected in any fold, which can occur at apical slices.
    """
    ys_all, xs_all = [], []
    for pred in preds_256:
        ys, xs = np.where(pred > 0)
        ys_all.extend(ys.tolist()); xs_all.extend(xs.tolist())

    if not ys_all:
        c = h_max // 2
        return (c - 80, c + 80, c - 80, c + 80)

    return (max(0,     min(ys_all) - margin),
            min(h_max, max(ys_all) + margin + 1),
            max(0,     min(xs_all) - margin),
            min(w_max, max(xs_all) + margin + 1))


def crop_to_bbox_and_resize(arr: np.ndarray, bbox: Tuple,
                             osz: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """Crop arr to bbox and rescale to output size osz.

    Bilinear interpolation (order=1) for images; nearest-neighbor (order=0) for masks
    to prevent label blending at class boundaries. Returns a zero array if the bbox
    is degenerate (zero area).
    """
    y0, y1, x0, x1 = bbox; cr = arr[y0:y1, x0:x1]
    if cr.size == 0:
        sh = osz + (arr.shape[-1],) if arr.ndim == 3 else osz
        return np.zeros(sh, dtype=arr.dtype)
    th, tw = osz; o = 0 if is_mask else 1
    if cr.ndim == 2: return zoom(cr, (th / cr.shape[0], tw / cr.shape[1]), order=o)
    return zoom(cr, (th / cr.shape[0], tw / cr.shape[1], 1), order=o)


def backproject_crop(pred_192: np.ndarray, bbox: Tuple,
                     H256: int = 256, W256: int = 256) -> np.ndarray:
    """Map a 192×192 crop-space prediction back to the 256×256 canvas.

    The prediction is rescaled to the original bounding box dimensions using
    nearest-neighbor interpolation (order=0), then written into a zero canvas
    at the bbox coordinates. Pixels outside the bbox remain background (0).
    """
    y0, y1, x0, x1 = bbox
    bh, bw = y1 - y0, x1 - x0
    canvas = np.zeros((H256, W256), dtype=pred_192.dtype)
    if bh <= 0 or bw <= 0: return canvas
    resized = zoom(pred_192.astype(np.float32),
                   (bh / pred_192.shape[0], bw / pred_192.shape[1]),
                   order=0).astype(pred_192.dtype)
    canvas[y0:y1, x0:x1] = resized
    return canvas

# ── 3D MORPHOLOGICAL CLEANUP ────────────────────────────────────────────────
def apply_3d_cca(vol: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Retain only the largest connected component per foreground class.

    Applying 3D connected component analysis over the stacked slice predictions
    removes spurious isolated islands (false positives) that can arise from
    uncertain apical slices or Stage-1 bbox misalignment. Background (class 0)
    is reconstructed implicitly from the surviving foreground labels.
    """
    out = np.zeros_like(vol)
    for c in range(1, num_classes):
        binary = (vol == c).astype(np.uint8)
        if binary.sum() == 0: continue
        labeled, n = nd_label(binary)
        if n == 0: continue
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        largest = np.argmax(sizes) + 1
        out[labeled == largest] = c
    return out

# ── PER-SLICE INFERENCE ─────────────────────────────────────────────────────
@torch.no_grad()
def predict_slice_s1(s1_models: List[nn.Module],
                     slice_3ch: np.ndarray) -> List[np.ndarray]:
    """Run Stage-1 inference on a single 2.5D slice and return one binary mask per fold.

    Each mask is thresholded at S1_BBOX_THRESHOLD; the union bounding box is computed
    from all masks afterward to obtain a robust localization for Stage 2.
    """
    t = torch.from_numpy(
        np.ascontiguousarray(slice_3ch.transpose(2, 0, 1))
    ).float().unsqueeze(0).to(device)

    preds = []
    for m in s1_models:
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            prob = torch.softmax(m(t), dim=1)[0, 1].cpu().numpy()
        preds.append((prob > S1_BBOX_THRESHOLD).astype(np.uint8))
    return preds


@torch.no_grad()
def predict_slice_s2(s2_models: List[nn.Module],
                     crop_3ch: np.ndarray) -> np.ndarray:
    """Run Stage-2 ensemble inference with Test-Time Augmentation on a cropped slice.

    4 TTA variants (original, h-flip, v-flip, both) × 5 fold models = 20 forward passes.
    Each augmented prediction is inverse-flipped before accumulation so all 20
    probability maps are aligned in the same coordinate frame before averaging.
    Returns mean softmax probabilities of shape (4, H, W).
    """
    accum = np.zeros((4, CROP_SIZE[0], CROP_SIZE[1]), dtype=np.float32)
    n_aug = 0

    for flip_h, flip_v in TTA_FLIPS:
        aug = crop_3ch.copy()
        if flip_h: aug = np.flip(aug, axis=0).copy()
        if flip_v: aug = np.flip(aug, axis=1).copy()

        t = torch.from_numpy(
            np.ascontiguousarray(aug.transpose(2, 0, 1))
        ).float().unsqueeze(0).to(device)

        fold_prob = np.zeros((4, CROP_SIZE[0], CROP_SIZE[1]), dtype=np.float32)
        for m in s2_models:
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = m(t)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            fold_prob += prob
        fold_prob /= len(s2_models)

        # invert the TTA flip before accumulating so all predictions are in the same spatial frame
        if flip_h: fold_prob = np.flip(fold_prob, axis=1).copy()
        if flip_v: fold_prob = np.flip(fold_prob, axis=2).copy()

        accum += fold_prob
        n_aug += 1

    return accum / n_aug

# ── VOLUME-LEVEL INFERENCE ──────────────────────────────────────────────────
def predict_volume(img_nii: nib.Nifti1Image,
                   s1_models: List[nn.Module],
                   s2_models: List[nn.Module]) -> np.ndarray:
    """Run the full two-stage inference pipeline on one NIfTI volume.

    Returns a segmentation volume with the same spatial shape as the input
    (H_orig × W_orig × Z), with integer class labels 0–3.
    """
    img_vol  = img_nii.get_fdata().astype(np.float32)
    orig_sp  = img_nii.header.get_zooms()[:3]
    H_orig, W_orig, Z = img_vol.shape

    img_res, (H_res, W_res) = preprocess_volume(img_vol, orig_sp)  # resampled volume + resampled spatial dims needed for back-projection

    pred_256_slices = []  # accumulate per-slice predictions in 256×256 canvas space

    for k in range(Z):
        # construct 2.5D input: stack adjacent slices as channels for inter-slice context
        prev = img_res[..., max(0, k - 1)]
        curr = img_res[..., k]
        nxt  = img_res[..., min(Z - 1, k + 1)]
        slice_3ch = pad_crop_2d(
            np.stack([prev, curr, nxt], axis=-1), TARGET_SIZE)  # (256, 256, 3)

        # Stage 1: localize the heart; union bbox across all folds for robustness
        s1_preds  = predict_slice_s1(s1_models, slice_3ch)
        bbox      = compute_heart_bbox_union(s1_preds)
        y0, y1, x0, x1 = bbox

        # crop to the localized region and rescale to the Stage-2 input resolution
        slice_256_1ch = slice_3ch
        crop_3ch = crop_to_bbox_and_resize(slice_256_1ch, bbox, CROP_SIZE, False)

        # Stage 2: produce averaged class probabilities over ensemble + TTA
        softmax_192 = predict_slice_s2(s2_models, crop_3ch)

        pred_192 = np.argmax(softmax_192, axis=0).astype(np.uint8)

        pred_256 = backproject_crop(pred_192, bbox)

        pred_256_slices.append(pred_256)

    # invert the pad/crop step to recover resampled-space coordinates
    pred_res = np.zeros((H_res, W_res, Z), dtype=np.uint8)
    for k, p256 in enumerate(pred_256_slices):
        pred_res[..., k] = invert_pad_crop_2d(p256, H_res, W_res)

    # resample back to original patient spacing so the prediction aligns with the input NIfTI geometry
    sx, sy = orig_sp[:2]
    zf_back = (H_orig / H_res, W_orig / W_res, 1.0)
    pred_orig = zoom(pred_res.astype(np.float32), zf_back, order=0).astype(np.uint8)

    # clamp to exact original shape; zoom may produce off-by-one sizes due to floating-point rounding
    pred_orig = pred_orig[:H_orig, :W_orig, :Z]

    pred_orig = apply_3d_cca(pred_orig, num_classes=4)

    return pred_orig

# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "XXXXXX" in str(EXP_DIR):
        raise ValueError("EXP_DIR still contains placeholder — set it to the training run directory before running.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR = ACDC_ROOT / "database" / "testing"

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_DIR}")

    print(f"\nLoading models from: {EXP_DIR}")
    s1_models = load_s1_models(EXP_DIR, K_FOLDS)
    s2_models = load_s2_models(EXP_DIR, K_FOLDS)

    test_patients = sorted([p for p in TEST_DIR.iterdir()
                             if p.is_dir() and p.name.startswith("patient")])
    print(f"Test patients: {len(test_patients)}")
    print(f"Output directory: {OUT_DIR}\n")

    for patient_dir in tqdm(test_patients, desc="Inference"):
        patient_id = patient_dir.name

        # each patient has two frames: end-diastole (ED) and end-systole (ES)
        nii_files = sorted([f for f in patient_dir.glob(f"{patient_id}_frame*.nii.gz")
                             if not f.name.endswith("_gt.nii.gz")])

        for img_path in nii_files:
            out_name = img_path.name.replace(".nii.gz", "_pred.nii.gz")
            out_path = OUT_DIR / out_name

            if out_path.exists():
                print(f"  Skipped (already exists): {out_name}")
                continue  # resume-safe: previously completed frames are not reprocessed

            img_nii = nib.load(str(img_path))
            pred    = predict_volume(img_nii, s1_models, s2_models)

            # preserve the original affine so the prediction is registered to the input image in any viewer
            pred_nii = nib.Nifti1Image(pred.astype(np.uint8), img_nii.affine)
            nib.save(pred_nii, str(out_path))

    print(f"\n✓ Done: {len(list(OUT_DIR.glob('*.nii.gz')))} prediction files written to {OUT_DIR}")
    print(f"\nNext step: run acdc_eval.py with")
    print(f"  PRED_DIR = '{OUT_DIR}'")
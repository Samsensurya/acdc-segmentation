#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════
ACDC Cardiac MRI Segmentation — Two-Stage Coarse-to-Fine Pipeline (5-Fold CV)
══════════════════════════════════════════════════════════════════════════════

Overview
--------
Cardiac MRI segmentation on the ACDC dataset (4 classes: background, right
ventricle, myocardium, left ventricle) using a two-stage cascaded approach:

  Stage 1 — Heart localization:
    A lightweight U-Net performs binary foreground/background segmentation on
    the full 256×256 image to localize the heart. Its predicted mask is used
    to compute a tight bounding box around the cardiac region.

  Stage 2 — Fine-grained segmentation:
    The region-of-interest defined by the Stage-1 bounding box is cropped and
    resized to 192×192. A deeper ResNet34-based Attention U-Net then performs
    4-class segmentation on this focused crop. Using the predicted bounding box
    (rather than the ground-truth) at both train and inference time ensures no
    information leakage and accurately reflects deployment conditions.

Input representation (2.5D)
----------------------------
Each 2D slice is passed as a 3-channel input: [previous slice, current slice,
next slice]. This pseudo-volumetric representation gives the model inter-slice
context without the memory cost of full 3D convolutions — an effective
trade-off for thin-slice cardiac MRI.

Training strategy
-----------------
- 5-fold cross-validation split at patient level to prevent data leakage.
- OneCycleLR for Stage 2 followed by Stochastic Weight Averaging (SWA), which
  averages model weights over late epochs to find a flatter loss minimum and
  improve generalization.
- Deep supervision: auxiliary segmentation heads at three decoder scales
  provide gradient signal deeper in the network, accelerating convergence.
- Loss: DiceFocal (overlap + hard-example mining) combined with an
  approximated Hausdorff distance term that penalizes boundary errors.

Resume
------
After the first run, hardcode RUN_NAME to resume from existing checkpoints.
Each fold saves its best checkpoint independently; completed folds are skipped.
"""

from __future__ import annotations
import os, random, time, json, re, copy, warnings
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, map_coordinates
import nibabel as nib
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.optim.swa_utils import AveragedModel, SWALR

import torchvision.models as tv_models
import monai
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# 0) SEED & DEVICE
# ══════════════════════════════════════════════════════════════════════════════
def set_global_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_global_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

USE_AMP     = device.type == "cuda"  # AMP (Automatic Mixed Precision): FP16 forward/backward, FP32 weight update; halves memory and speeds up on Tensor Cores
USE_COMPILE = False   # torch.compile wraps the model and renames state_dict keys with an _orig_mod. prefix, which breaks checkpoint loading on resume; disabled for compatibility

if device.type == "cuda":
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

print(f"Device: {device} | AMP: {USE_AMP} | MONAI {monai.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
# 1) CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
ACDC_ROOT = Path("/content/drive/MyDrive/ACDC")
OUT_2D_V2 = ACDC_ROOT / "database" / "prepared_2d_v2"

TARGET_SPACING_XY = (1.37, 1.37)
TARGET_SIZE       = (256, 256)
CROP_SIZE         = (192, 192)
CROP_MARGIN       = 40    # padding around the predicted bounding box; wider margin compensates for localization uncertainty when using Stage-1 predictions

K_FOLDS = 5
SEED    = 42

S1_BATCH = 32;  S1_LR = 1e-3;  S1_EPOCHS = 160; S1_PAT = 20
S2_BATCH = 32;  S2_LR = 3e-4;  S2_EPOCHS = 250; S2_PAT = 40  # Stage 2: finer task, lower LR, longer schedule
S2_WD    = 1e-4; S2_ENC_LR_MULT = 0.1

SWA_START_FRAC    = 0.35   # SWA begins after 35% of total epochs; model must be past initial convergence but still averaging diverse loss-landscape solutions
SWA_LR            = 5e-5    # constant LR during SWA phase; low value keeps the model near the converged basin
DS_WEIGHTS        = [0.125, 0.25, 0.5, 1.0]  # deep supervision loss weights per scale (coarsest to finest); main output is dominant
LAMBDA_HD         = 0.2    # weight for the Hausdorff distance loss term; encourages accurate delineation of structure boundaries

S2_BBOX_STRATEGY  = "s1_only"  # use Stage-1 predicted boxes at train time; matches inference and avoids GT leakage into Stage-2 crops
S1_BBOX_THRESHOLD = 0.4  # probability threshold for Stage-1 binary mask; lower value favors recall, producing larger bounding boxes
MYO_CLASS_WEIGHT  = torch.tensor([1.0, 1.0, 1.0, 1.0])  # per-class loss weights; uniform to avoid systematic bias in predicted myocardial mass

USE_INSTANCE_NORM = False
USE_RAM_CACHE     = True
NUM_WORKERS = 2 if USE_RAM_CACHE else 4
PIN_MEMORY  = device.type == "cuda"

print(f"\nConfig:")
print(f"  BBox: {S2_BBOX_STRATEGY} | Threshold: {S1_BBOX_THRESHOLD} | Margin: {CROP_MARGIN}")
print(f"  MYO_W: {MYO_CLASS_WEIGHT[2]:.1f} | S2_BATCH: {S2_BATCH} | LAMBDA_HD: {LAMBDA_HD} | pct_start: 0.10")
print(f"  SWA: start={SWA_START_FRAC} lr={SWA_LR}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2) EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def resample_volume_xy(img_vol, msk_vol, orig_spacing, target_spacing=(1.37, 1.37)):
    sx, sy, _ = orig_spacing[:3]
    zf = (sx / target_spacing[0], sy / target_spacing[1], 1.0)
    return (zoom(img_vol, zf, order=1).astype(np.float32),
            zoom(msk_vol, zf, order=0).astype(np.uint8))


def export_acdc_v2(acdc_root, out_root, target_spacing_xy=(1.37, 1.37), seed=42):
    acdc_root = Path(acdc_root); raw_root = acdc_root / "database" / "training"
    out_root  = Path(out_root)
    if not raw_root.exists(): raise FileNotFoundError(f"Raw ACDC not found: {raw_root}")
    patients = sorted([p for p in raw_root.iterdir()
                       if p.is_dir() and p.name.startswith("patient")])
    (out_root / "all" / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "all" / "masks").mkdir(parents=True, exist_ok=True)
    total = 0
    for patient_dir in tqdm(patients, desc="Export v2"):
        nii_files = sorted(patient_dir.glob(f"{patient_dir.name}_frame*.nii.gz"))
        img_files = [f for f in nii_files if not f.name.endswith("_gt.nii.gz")]
        for img_path in img_files:
            mask_path = img_path.with_name(img_path.name.replace(".nii.gz", "_gt.nii.gz"))
            if not mask_path.exists(): continue
            img_nii = nib.load(str(img_path)); msk_nii = nib.load(str(mask_path))
            img_vol = img_nii.get_fdata().astype(np.float32)
            msk_vol = msk_nii.get_fdata().astype(np.uint8)
            orig_sp = img_nii.header.get_zooms()[:3]
            lo, hi  = np.percentile(img_vol, 1.0), np.percentile(img_vol, 99.0)
            img_vol = np.clip(img_vol, lo, hi)
            mu, sig = img_vol.mean(), img_vol.std()
            img_vol = ((img_vol - mu) / sig).astype(np.float32) if sig > 1e-8 \
                      else np.zeros_like(img_vol)
            img_vol, msk_vol = resample_volume_xy(img_vol, msk_vol, orig_sp, target_spacing_xy)
            for k in range(img_vol.shape[-1]):
                name = img_path.name.replace(".nii.gz", f"_slice{k:03d}.npy")
                np.save(out_root / "all" / "images" / name, img_vol[..., k])
                np.save(out_root / "all" / "masks"  / name, msk_vol[..., k])
                total += 1
    print(f"Export complete: {total} slices saved")


if not (OUT_2D_V2 / "all" / "images").exists():
    export_acdc_v2(ACDC_ROOT, OUT_2D_V2, TARGET_SPACING_XY, SEED)
else:
    n = len(list((OUT_2D_V2 / "all" / "images").glob("*.npy")))
    print(f"Export already exists ({n} slices found)")

# ══════════════════════════════════════════════════════════════════════════════
# 3) DICT-BUILDING + K-FOLD
# ══════════════════════════════════════════════════════════════════════════════
def extract_patient(name: str) -> str:
    m = re.match(r"(patient\d+)", name); return m.group(1) if m else name


def build_all_dicts_25d(out_root: Path) -> List[Dict]:
    img_dir     = out_root / "all" / "images"
    msk_dir     = out_root / "all" / "masks"
    img_paths   = sorted(img_dir.glob("*.npy"))
    msk_by_name = {p.name: p for p in sorted(msk_dir.glob("*.npy"))}
    def vol_prefix(n): return re.sub(r"_slice\d+\.npy$", "", n)
    volumes = defaultdict(list)
    for p in img_paths: volumes[vol_prefix(p.name)].append(p)
    dicts = []
    for prefix, slices in volumes.items():
        slices = sorted(slices); n = len(slices)
        for i, curr in enumerate(slices):
            prev = slices[max(0, i - 1)]; nxt = slices[min(n - 1, i + 1)]
            msk  = msk_by_name.get(curr.name)
            if msk is None: continue
            dicts.append({"img_prev": str(prev), "img_curr": str(curr),
                          "img_next": str(nxt),  "mask": str(msk),
                          "patient":  extract_patient(curr.name)})
    return dicts


def kfold_patient_split(all_dicts, k=5, seed=42):
    patients  = sorted(set(d["patient"] for d in all_dicts))
    rng = random.Random(seed); rng.shuffle(patients)
    fold_size = len(patients) // k
    folds = [set(patients[i * fold_size : (i+1) * fold_size if i < k-1 else len(patients)])
             for i in range(k)]
    for fi in range(k):
        val_p = folds[fi]
        yield fi, [d for d in all_dicts if d["patient"] not in val_p], \
                  [d for d in all_dicts if d["patient"] in val_p]


all_dicts = build_all_dicts_25d(OUT_2D_V2)
patients  = sorted(set(d["patient"] for d in all_dicts))
print(f"Total: {len(all_dicts)} slices from {len(patients)} patients")

# ══════════════════════════════════════════════════════════════════════════════
# 4) RAM CACHE
# ══════════════════════════════════════════════════════════════════════════════
class NpyCache:
    def __init__(self):
        self._data: Dict[str, np.ndarray] = {}
        self._hits = 0; self._misses = 0

    def get(self, path: str) -> Optional[np.ndarray]:
        if path in self._data:
            self._hits += 1; return self._data[path]
        try:
            arr = np.load(path); self._data[path] = arr
            self._misses += 1; return arr
        except (EOFError, ValueError, OSError): return None

    def preload(self, paths: List[str], desc: str = "Cache"):
        for path in tqdm(paths, desc=f"  {desc}", leave=False): self.get(path)
        mb = sum(a.nbytes for a in self._data.values()) / 1024 / 1024
        print(f"  Cache: {len(self._data)} Arrays | {mb:.0f} MB RAM")

    def stats(self) -> str:
        total = self._hits + self._misses
        rate  = self._hits / total * 100 if total > 0 else 0
        return f"Cache: {len(self._data)} Arrays | Hit-Rate: {rate:.1f}%"


_GLOBAL_CACHE = NpyCache() if USE_RAM_CACHE else None


def safe_load_npy(path: str) -> Optional[np.ndarray]:
    if _GLOBAL_CACHE is not None: return _GLOBAL_CACHE.get(path)
    try: return np.load(path)
    except (EOFError, ValueError, OSError): return None


def preload_all_data(all_dicts: List[Dict]):
    if not USE_RAM_CACHE: return
    print("\n  Loading all data into RAM cache...")
    all_paths = {d[k] for d in all_dicts for k in ["img_prev","img_curr","img_next","mask"]}
    _GLOBAL_CACHE.preload(sorted(all_paths), "ACDC → RAM cache")

# ══════════════════════════════════════════════════════════════════════════════
# 5) HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def pad_crop_2d(arr, ts):
    th, tw = ts; h, w = arr.shape[:2]
    ph = max(0, th - h); pw = max(0, tw - w)
    if arr.ndim == 2:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    else:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    h, w = arr.shape[:2]
    return arr[(h - th) // 2:(h - th) // 2 + th, (w - tw) // 2:(w - tw) // 2 + tw]


def compute_heart_bbox(mask, margin=40):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape; c = h // 2
        return (c - 80, c + 80, c - 80, c + 80)
    h, w = mask.shape
    return (max(0, ys.min() - margin),      min(h, ys.max() + margin + 1),
            max(0, xs.min() - margin),      min(w, xs.max() + margin + 1))


def crop_to_bbox_and_resize(arr, bbox, osz, is_mask=False):
    y0, y1, x0, x1 = bbox; cr = arr[y0:y1, x0:x1]
    if cr.size == 0:
        sh = osz + (arr.shape[-1],) if arr.ndim == 3 else osz
        return np.zeros(sh, dtype=arr.dtype)
    th, tw = osz; o = 0 if is_mask else 1
    if cr.ndim == 2: return zoom(cr, (th / cr.shape[0], tw / cr.shape[1]), order=o)
    return zoom(cr, (th / cr.shape[0], tw / cr.shape[1], 1), order=o)


def augment_geometric(img, mask):
    if random.random() < 0.5: img = np.flip(img, 1).copy(); mask = np.flip(mask, 1).copy()
    if random.random() < 0.5: img = np.flip(img, 0).copy(); mask = np.flip(mask, 0).copy()
    k = random.randint(0, 3)
    if k: img = np.rot90(img, k, (0, 1)).copy(); mask = np.rot90(mask, k, (0, 1)).copy()
    return img, mask


def augment_elastic(img, mask, alpha=800, sigma=30):
    h, w = img.shape[:2]
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    if img.ndim == 3:
        out = np.empty_like(img)
        for c in range(img.shape[-1]):
            out[..., c] = map_coordinates(img[..., c], coords, order=1, mode="reflect")
        img = out
    else:
        img = map_coordinates(img, coords, order=1, mode="reflect")
    mask = map_coordinates(
        mask.astype(np.float32), coords, order=0, mode="reflect").astype(np.int64)
    return img, mask


def augment_intensity(img):
    if random.random() < 0.2: img = img + np.random.uniform(-0.1, 0.1)
    if random.random() < 0.2:
        f = np.random.uniform(0.9, 1.1)
        for c in range(img.shape[-1]):
            m = img[..., c].mean(); img[..., c] = (img[..., c] - m) * f + m
    if random.random() < 0.15:
        img = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, -5.0, 5.0)  # clip to prevent augmented values from destabilizing BatchNorm statistics in downstream layers


def get_loader_kwargs(train: bool) -> dict:
    kwargs = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                  drop_last=train, shuffle=train)
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True; kwargs["prefetch_factor"] = 2
    return kwargs

# ══════════════════════════════════════════════════════════════════════════════
# 6) STAGE-1 BBOX INFERENZ
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_s1_bboxes(s1_model: nn.Module, dicts: List[Dict],
                      margin: int = CROP_MARGIN,
                      threshold: float = S1_BBOX_THRESHOLD) -> Dict[str, tuple]:
    s1_model.eval(); bboxes = {}
    loader = monai.data.DataLoader(
        _BBoxInferDataset(dicts, TARGET_SIZE),
        batch_size=64, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    for batch in tqdm(loader, desc="  S1 BBox Inference", leave=False):
        imgs  = batch["img"].to(device, non_blocking=True)
        probs = torch.softmax(s1_model(imgs), dim=1)[:, 1]
        preds = (probs > threshold).cpu().numpy()
        for i, path in enumerate(batch["path"]):
            bboxes[path] = compute_heart_bbox(preds[i].astype(np.uint8), margin)
    s1_model.train(); return bboxes


class _BBoxInferDataset(TorchDataset):
    def __init__(self, dicts, target_size):
        self.dicts = dicts; self.ts = target_size

    def __len__(self): return len(self.dicts)

    def __getitem__(self, idx):
        d = self.dicts[idx]
        arrs = [safe_load_npy(d[k]) for k in ["img_prev", "img_curr", "img_next"]]
        if any(a is None for a in arrs): return self.__getitem__((idx + 1) % len(self))
        img = np.stack([a.astype(np.float32) for a in arrs], axis=-1)
        img = pad_crop_2d(img, self.ts)
        return {"img":  torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
                "path": d["img_curr"]}

# ══════════════════════════════════════════════════════════════════════════════
# 7) DATASETS
# ══════════════════════════════════════════════════════════════════════════════
class BinaryHeartDataset(TorchDataset):
    def __init__(self, dicts, target_size=(256, 256), augment=False):
        self.dicts = dicts; self.ts = target_size; self.aug = augment

    def __len__(self): return len(self.dicts)

    def __getitem__(self, idx):
        d = self.dicts[idx]
        arrs = [safe_load_npy(d[k]) for k in ["img_prev", "img_curr", "img_next", "mask"]]
        if any(a is None for a in arrs): return self.__getitem__((idx + 1) % len(self))
        img  = np.stack([a.astype(np.float32) for a in arrs[:3]], axis=-1)
        mask = (arrs[3] > 0).astype(np.int64)
        img  = pad_crop_2d(img, self.ts); mask = pad_crop_2d(mask, self.ts)
        if self.aug:
            img, mask = augment_geometric(img, mask)
            if random.random() < 0.3:
                img, mask = augment_elastic(img, mask,
                    np.random.uniform(400, 1000), np.random.uniform(20, 40))
            img = augment_intensity(img)
        return {
            "img":  torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long().unsqueeze(0),
        }


class CroppedHeartDataset(TorchDataset):
    def __init__(self, dicts, crop_size=(192, 192), margin=40, augment=False,
                 strategy="s1_only", s1_bboxes=None):
        self.dicts     = dicts; self.cs = crop_size
        self.margin    = margin; self.aug = augment
        self.strategy  = strategy; self.s1_bboxes = s1_bboxes or {}

    def __len__(self): return len(self.dicts)

    def _get_bbox(self, d, mask):
        gt_bbox = compute_heart_bbox((mask > 0).astype(np.uint8), self.margin)
        s1_bbox = self.s1_bboxes.get(d["img_curr"])
        if self.strategy == "gt_only" or s1_bbox is None: use_s1 = False
        elif self.strategy == "s1_only": use_s1 = True
        else: use_s1 = random.random() < 0.5
        if use_s1: return s1_bbox
        if self.aug:
            y0, y1, x0, x1 = gt_bbox; j = 20  # random jitter on GT box: trains robustness to slight crop misalignments
            return (max(0,              y0 + random.randint(-j, j)),
                    min(TARGET_SIZE[0], y1 + random.randint(-j, j)),
                    max(0,              x0 + random.randint(-j, j)),
                    min(TARGET_SIZE[1], x1 + random.randint(-j, j)))
        return gt_bbox

    def __getitem__(self, idx):
        d = self.dicts[idx]
        arrs = [safe_load_npy(d[k]) for k in ["img_prev", "img_curr", "img_next", "mask"]]
        if any(a is None for a in arrs): return self.__getitem__((idx + 1) % len(self))
        img  = np.stack([a.astype(np.float32) for a in arrs[:3]], axis=-1)
        mask = arrs[3].astype(np.int64)
        img  = pad_crop_2d(img, TARGET_SIZE); mask = pad_crop_2d(mask, TARGET_SIZE)
        bbox = self._get_bbox(d, mask)
        img_c  = crop_to_bbox_and_resize(img,  bbox, self.cs, False)
        mask_c = np.round(crop_to_bbox_and_resize(mask, bbox, self.cs, True)).astype(np.int64)
        if self.aug:
            img_c, mask_c = augment_geometric(img_c, mask_c)
            if random.random() < 0.3:
                img_c, mask_c = augment_elastic(img_c, mask_c,
                    np.random.uniform(300, 700), np.random.uniform(15, 30))
            img_c = augment_intensity(img_c)
        return {
            "img":  torch.from_numpy(np.ascontiguousarray(img_c.transpose(2, 0, 1))).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_c)).long().unsqueeze(0),
        }

# ══════════════════════════════════════════════════════════════════════════════
# 8) MODEL
# ══════════════════════════════════════════════════════════════════════════════
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
    def __init__(self, num_classes=4, pretrained=True, dropout=0.1):
        super().__init__()
        r = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None)
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
        main = self.final_conv(out)
        if self.training:
            return (main,
                    F.interpolate(self.ds_head4(d4), sz, mode="bilinear", align_corners=False),
                    F.interpolate(self.ds_head3(d3), sz, mode="bilinear", align_corners=False),
                    F.interpolate(self.ds_head2(d2), sz, mode="bilinear", align_corners=False))
        return main

    def encoder_params(self):
        return (list(self.enc0.parameters()) + list(self.enc1.parameters()) +
                list(self.enc2.parameters()) + list(self.enc3.parameters()) +
                list(self.enc4.parameters()))

    def decoder_params(self):
        enc_ids = {id(p) for p in self.encoder_params()}
        return [p for p in self.parameters() if id(p) not in enc_ids]


_t = ResNetAttentionUNet_DS(4, True).to(device); _t.train()
with torch.no_grad(): _o = _t(torch.randn(2, 3, 192, 192).to(device))
print(f"Model OK | main: {_o[0].shape}")
del _t

# ══════════════════════════════════════════════════════════════════════════════
# 9) LOSS
# ══════════════════════════════════════════════════════════════════════════════
def hausdorff_dt_loss_cpu(probs: torch.Tensor,
                          masks: torch.Tensor,
                          nc: int = 4) -> torch.Tensor:
    """
    Hausdorff Distance Loss via Euclidean Distance Transform (CPU/scipy).
    No CuPy or NVRTC dependency — works on any CUDA version.

    Method (Karimi & Salcudean 2019):
      1. Move softmax probs and one-hot GT to CPU
      2. Compute Euclidean Distance Transform of GT mask → D_gt
      3. Loss = mean( (p - gt)² × D_gt² )
      → Large penalty for predictions far from the true boundary
      → Gradient flows through p; differentiable w.r.t. probabilities

    Applied to the main output only — not to deep supervision heads.
    """
    from scipy.ndimage import distance_transform_edt as edt
    probs_np = probs.detach().cpu().float().numpy()   # (B, C, H, W)
    masks_np = masks.cpu().numpy()                     # (B, 1, H, W)
    B = probs_np.shape[0]
    total = 0.0

    for b in range(B):
        gt_b = masks_np[b, 0]   # (H, W)
        for c in range(1, nc):  # skip background class
            gt_c  = (gt_b == c).astype(np.float32)
            # Euclidean distance transforms of GT mask and its complement
            dt_fg = edt(gt_c).astype(np.float32)        # EDT outward from foreground region
            dt_bg = edt(1 - gt_c).astype(np.float32)    # EDT inward from background region
            dt    = (dt_fg + dt_bg).astype(np.float32)  # summed: large where predictions deviate far from boundary
            p_c   = probs_np[b, c]                       # predicted probability for class c, shape (H, W)
            total += float(np.mean((p_c - gt_c) ** 2 * dt ** 2))

    # normalize by total number of class-batch pairs
    n = B * (nc - 1)
    return torch.tensor(total / max(n, 1), dtype=torch.float32,
                        requires_grad=False, device=probs.device)


class CombinedDeepSupLoss(nn.Module):
    def __init__(self, num_classes=4, ds_weights=(0.125, 0.25, 0.5, 1.0),
                 class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice_focal = monai.losses.DiceFocalLoss(
            to_onehot_y=True, softmax=True,
            lambda_dice=1.0, lambda_focal=1.0, gamma=2.0,
            weight=class_weight)
        self.ds_weights  = ds_weights
        self.num_classes = num_classes

    def _single(self, logits, masks):
        loss = self.dice_focal(logits, masks)
        if LAMBDA_HD > 0:
            # softmax probs required for HD loss; runs on CPU via scipy — no CuPy dependency
            probs = torch.softmax(logits, dim=1)
            hd    = hausdorff_dt_loss_cpu(probs, masks, self.num_classes)
            loss  = loss + LAMBDA_HD * hd.to(loss.device)
        return loss

    def forward(self, outputs, masks):
        if isinstance(outputs, tuple):
            main, ds4, ds3, ds2 = outputs
            # HD loss on main output only — too expensive for all DS heads; gradient signal sufficient
            dice_only = self.dice_focal
            main_loss = self._single(main, masks)
            ds_loss   = (self.ds_weights[0] * dice_only(ds4,  masks)
                       + self.ds_weights[1] * dice_only(ds3,  masks)
                       + self.ds_weights[2] * dice_only(ds2,  masks))
            total = self.ds_weights[3] * main_loss + ds_loss
            return total / sum(self.ds_weights)
        return self._single(outputs, masks)

# ══════════════════════════════════════════════════════════════════════════════
# 10) TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def save_ckpt(path, model, opt, sched, epoch, best_dice, extra=None):
    state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}  # strip _orig_mod. prefix so checkpoint loads cleanly into a plain model
    d = {"epoch": epoch, "model_state_dict": state,
         "optimizer_state_dict": opt.state_dict(), "best_dice": best_dice}
    if sched: d["scheduler_state_dict"] = sched.state_dict()
    if extra: d.update(extra)
    torch.save(d, str(path))


def train_epoch(model, loader, loss_fn, optimizer, scheduler, device, scaler):
    model.train(); total = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        imgs  = batch["img"].to(device, non_blocking=True)   # non_blocking overlaps host-to-device copy with worker CPU preprocessing
        masks = batch["mask"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):    # FP16 autocast; GradScaler prevents gradient underflow
            out = model(imgs); loss = loss_fn(out, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def val_epoch_classwise(model, loader, loss_fn, device, nc=4):
    model.eval(); total = 0.0
    names   = {1: "RV", 2: "MYO", 3: "LV"}
    metrics = {c: monai.metrics.DiceMetric(include_background=False, reduction="mean")
               for c in range(1, nc)}
    for batch in tqdm(loader, desc="  Val", leave=False):
        imgs  = batch["img"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).long()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(imgs); total += loss_fn(logits, masks).item()
        pred     = torch.argmax(torch.softmax(logits, 1), 1)
        pred_oh  = monai.networks.one_hot(pred.unsqueeze(1), nc)
        masks_oh = monai.networks.one_hot(masks.squeeze(1).unsqueeze(1), nc)
        for c in range(1, nc):
            metrics[c](y_pred=pred_oh[:, c:c+1], y=masks_oh[:, c:c+1])
    class_dice = {names[c]: float(metrics[c].aggregate().mean()) for c in range(1, nc)}
    return total / max(1, len(loader)), float(np.mean(list(class_dice.values()))), class_dice


def update_bn_swa(swa_model, loader, device):
    """Recompute BatchNorm running statistics for the SWA-averaged model.

    SWA averages weights but not BN running statistics, which become inconsistent
    with the averaged weights. A full forward pass over the training data recomputes
    correct running_mean and running_var. running_var is initialized to 1 (not 0)
    and momentum is set to None so PyTorch uses a cumulative moving average,
    equivalent to the exact batch statistics over the full training set.
    """
    momenta = {}
    for m in swa_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.running_mean.zero_(); m.running_var.fill_(1)
            momenta[m] = m.momentum; m.momentum = None
            m.num_batches_tracked.fill_(1)
    swa_model.train()
    with torch.no_grad():
        for batch in loader: swa_model(batch["img"].to(device, non_blocking=True))
    for bn, mom in momenta.items(): bn.momentum = mom
    swa_model.eval()


def _extract_swa_state(swa_model) -> dict:
    """Extract a clean state_dict from an AveragedModel wrapper.

    AveragedModel prepends module. to all keys. If the wrapped model was also
    compiled with torch.compile, _orig_mod. is additionally prepended. Both
    prefixes are stripped so the dict loads cleanly into a plain model.
    The internal n_averaged counter is excluded as it is not part of model state.
    """
    raw = {k: v for k, v in swa_model.state_dict().items() if k != "n_averaged"}
    return {k.removeprefix("module.").replace("_orig_mod.", ""): v for k, v in raw.items()}


def train_stage(model, tl, vl, loss_fn, opt, sched, device, exp_dir,
                max_ep, patience, nc, name,
                use_swa=False, swa_start_ep=None, swa_lr=1e-5):
    """
    Train one stage (Stage 1 or Stage 2) with optional SWA and early stopping.
    Model is passed in without torch.compile so checkpoints can be loaded cleanly
    without key prefix mismatches.
    """
    ckpt_dir  = exp_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_{name}.pt"

    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [{name}] SKIPPED (checkpoint already exists, Dice={ckpt['best_dice']:.4f})")
        return model, ckpt["best_dice"]

    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    swa_model = None; swa_sched = None
    if use_swa and swa_start_ep:
        swa_model = AveragedModel(model, device=device)
        swa_sched = SWALR(opt, swa_lr=swa_lr)

    best_dice = -1.0; no_imp = 0; hist = []
    is_s2 = (nc == 4)

    for ep in range(1, max_ep + 1):
        t0 = time.time()
        swa_active = swa_model is not None and ep >= swa_start_ep

        if swa_active:
            tl_loss = train_epoch(model, tl, loss_fn, opt, None, device, scaler)
            swa_model.update_parameters(model); swa_sched.step()
        else:
            tl_loss = train_epoch(model, tl, loss_fn, opt, sched, device, scaler)

        if is_s2:
            vl_loss, vd, class_dice = val_epoch_classwise(model, vl, loss_fn, device, nc)
            cd_str = (f"RV={class_dice['RV']:.3f} "
                      f"MYO={class_dice['MYO']:.3f} "
                      f"LV={class_dice['LV']:.3f}")
        else:
            dm = monai.metrics.DiceMetric(include_background=False, reduction="mean"); dm.reset()
            model.eval(); vl_loss = 0.0
            with torch.no_grad():
                for batch in vl:
                    imgs  = batch["img"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True).long()
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        logits = model(imgs); vl_loss += loss_fn(logits, masks).item()
                    pred = torch.argmax(torch.softmax(logits, 1), 1)
                    dm(y_pred=monai.networks.one_hot(pred.unsqueeze(1), nc),
                       y=monai.networks.one_hot(masks.squeeze(1).unsqueeze(1), nc))
            vl_loss /= max(1, len(vl)); vd = float(dm.aggregate().item())
            class_dice = {}; cd_str = ""

        dt = time.time() - t0; lr = opt.param_groups[0]["lr"]
        tag = " [SWA]" if swa_active else ""
        row = {"epoch": ep, "train_loss": tl_loss, "val_loss": vl_loss,
               "val_dice": vd, "lr": lr, "time_s": dt}
        row.update({f"dice_{k}": v for k, v in class_dice.items()})
        hist.append(row)

        print(f"  [{name}] Ep {ep:03d}/{max_ep} | "
              f"t={tl_loss:.4f} v={vl_loss:.4f} dice={vd:.4f} "
              f"lr={lr:.2e} {dt:.0f}s{tag}")
        if cd_str: print(f"         {cd_str}")

        if vd > best_dice:
            best_dice = vd; no_imp = 0
            save_ckpt(best_ckpt, model, opt, sched, ep, best_dice)
            print(f"    ✓ Best: {best_dice:.4f}")
        else:
            no_imp += 1

        pd.DataFrame(hist).to_csv(exp_dir / f"history_{name}.csv", index=False)

        # Early stopping only applies before SWA; once SWA is active the model continues
        # regardless of validation plateau to allow weight averaging to converge
        if no_imp >= patience and not swa_active:
            print("    Early stop."); break

    if swa_model:
        print(f"  [{name}] Updating SWA batch norm statistics...")
        update_bn_swa(swa_model, tl, device)
        if is_s2:
            _, vd_swa, cd_swa = val_epoch_classwise(swa_model, vl, loss_fn, device, nc)
            print(f"  SWA: {vd_swa:.4f} "
                  f"(RV={cd_swa['RV']:.3f} MYO={cd_swa['MYO']:.3f} "
                  f"LV={cd_swa['LV']:.3f}) vs Best: {best_dice:.4f}")
        else:
            dm = monai.metrics.DiceMetric(include_background=False, reduction="mean"); dm.reset()
            swa_model.eval()
            with torch.no_grad():
                for batch in vl:
                    pred = torch.argmax(torch.softmax(
                        swa_model(batch["img"].to(device)), 1), 1)
                    dm(y_pred=monai.networks.one_hot(pred.unsqueeze(1), nc),
                       y=monai.networks.one_hot(
                           batch["mask"].to(device).squeeze(1).unsqueeze(1).long(), nc))
            vd_swa = float(dm.aggregate().item())
            print(f"  SWA Dice: {vd_swa:.4f} vs Best: {best_dice:.4f}")

        if vd_swa > best_dice:
            best_dice = vd_swa
            model.load_state_dict(_extract_swa_state(swa_model), strict=False)  # transfer averaged weights back into the base model
            save_ckpt(best_ckpt, model, opt, sched, ep, best_dice, {"swa": True})
            print(f"    ✓ SWA improved: {best_dice:.4f}")
        else:
            ckpt = torch.load(str(best_ckpt), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"  [{name}] Done. Best Dice: {best_dice:.4f}\n")
    return model, best_dice

# ══════════════════════════════════════════════════════════════════════════════
# 11) K-FOLD TRAINING
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Unique identifier for this experiment run. All checkpoints, history CSVs,
    # and config files are written under ACDC_ROOT/runs/RUN_NAME/.
    # Set to None on the first run (auto-generates a timestamped name);
    # hardcode the generated name here before resuming an interrupted run.
    # RUN_NAME = "last_run_20260331_115106"
    RUN_NAME = None   # auto-generate: "last_run_YYYYMMDD_HHMMSS"

    if RUN_NAME is None:
        RUN_NAME = "last_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    EXP_DIR = ACDC_ROOT / "runs" / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment":        "final_overnight",
        "s2_bbox_strategy":  S2_BBOX_STRATEGY,
        "s1_bbox_threshold": S1_BBOX_THRESHOLD,
        "crop_margin":       CROP_MARGIN,
        "myo_weight":        float(MYO_CLASS_WEIGHT[2]),
        "s2_batch":          S2_BATCH,
        "pct_start":         0.10,
        "swa_start_frac":    SWA_START_FRAC,   # fraction of S2_EPOCHS before SWA begins
        "swa_lr":            SWA_LR,
        "k_folds":           K_FOLDS,
        "s2_epochs":         S2_EPOCHS,
        "use_compile":       USE_COMPILE,
        "notes": [
            "LAMBDA_HD = 0.2: Hausdorff distance loss added to improve boundary delineation",
            "MYO_CLASS_WEIGHT = 1.0: uniform weighting avoids systematic myocardial volume bias",
            "S2_BATCH = 32: larger batch reduces gradient variance ahead of SWA averaging",
        ],
    }
    with open(EXP_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  LAST RUN — 5-Fold Training")
    print(f"  RUN: {EXP_DIR}")
    print(f"  BBox: {S2_BBOX_STRATEGY} | Threshold: {S1_BBOX_THRESHOLD} | Margin: {CROP_MARGIN}")
    print(f"  MYO_W: {MYO_CLASS_WEIGHT[2]:.1f} | S2_BATCH: {S2_BATCH} | LAMBDA_HD: {LAMBDA_HD} | pct_start: 0.10")
    print(f"{'=' * 70}\n")

    preload_all_data(all_dicts)
    if _GLOBAL_CACHE: print(f"  {_GLOBAL_CACHE.stats()}\n")

    all_s1 = []; all_s2 = []; fold_results = []

    for fi, train_d, val_d in kfold_patient_split(all_dicts, K_FOLDS, SEED):
        fold_dir = EXP_DIR / f"fold_{fi}"; fold_dir.mkdir(exist_ok=True)
        nt = len(set(d["patient"] for d in train_d))
        nv = len(set(d["patient"] for d in val_d))
        print(f"\n{'─' * 70}")
        print(f"  FOLD {fi + 1}/{K_FOLDS} | Train: {len(train_d)} ({nt}p) | Val: {len(val_d)} ({nv}p)")
        print(f"{'─' * 70}")

        # ── Stage 1: Heart localization ──────────────────────────────────────
        # Lightweight UNet (no pretrained encoder) trained for binary segmentation.
        # Compact channel counts (32→256) are sufficient for the coarse localization task.
        print(f"\n  ── Stage 1 ──")
        s1 = monai.networks.nets.UNet(
            spatial_dims=2, in_channels=3, out_channels=2,
            channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=1
        ).to(device)

        s1_tl = monai.data.DataLoader(BinaryHeartDataset(train_d, TARGET_SIZE, True),
                                       batch_size=S1_BATCH, **get_loader_kwargs(True))
        s1_vl = monai.data.DataLoader(BinaryHeartDataset(val_d,   TARGET_SIZE, False),
                                       batch_size=S1_BATCH, **get_loader_kwargs(False))
        s1_loss = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)  # combined Dice + Cross-Entropy
        s1_opt  = torch.optim.AdamW(s1.parameters(), lr=S1_LR, weight_decay=1e-4)
        s1_sch  = torch.optim.lr_scheduler.CosineAnnealingLR(s1_opt, S1_EPOCHS, 1e-6)  # cosine decay to near-zero LR

        s1, s1d = train_stage(s1, s1_tl, s1_vl, s1_loss, s1_opt, s1_sch, device,
                               fold_dir, S1_EPOCHS, S1_PAT, nc=2, name=f"s1_f{fi}")
        all_s1.append(copy.deepcopy(s1.cpu())); s1.to(device)  # keep a CPU copy for ensemble inference

        # ── Inter-stage: derive bounding boxes from Stage-1 predictions ───────
        # Run inference over train and val splits to produce per-slice bbox dicts.
        # These are passed to Stage-2 datasets as localization priors.
        print(f"\n  ── S1 BBox Inference (threshold={S1_BBOX_THRESHOLD}, margin={CROP_MARGIN}) ──")
        s1_bboxes_train = predict_s1_bboxes(s1, train_d)
        s1_bboxes_val   = predict_s1_bboxes(s1, val_d)
        print(f"  Train: {len(s1_bboxes_train)} | Val: {len(s1_bboxes_val)}")

        # ── Stage 2: Fine-grained 4-class segmentation ───────────────────────
        # ResNet34 encoder pretrained on ImageNet; decoder trained from scratch.
        # Crops are defined by Stage-1 bounding boxes (strategy="s1_only").
        print(f"\n  ── Stage 2 [s1_only | Margin={CROP_MARGIN} | Batch={S2_BATCH}] ──")
        s2 = ResNetAttentionUNet_DS(4, True, 0.1).to(device)

        s2_tds = CroppedHeartDataset(train_d, CROP_SIZE, CROP_MARGIN, augment=True,
                                      strategy=S2_BBOX_STRATEGY,
                                      s1_bboxes=s1_bboxes_train)
        s2_vds = CroppedHeartDataset(val_d, CROP_SIZE, CROP_MARGIN, augment=False,
                                      strategy="s1_only",
                                      s1_bboxes=s1_bboxes_val)

        s2_tl = monai.data.DataLoader(s2_tds, batch_size=S2_BATCH, **get_loader_kwargs(True))
        s2_vl = monai.data.DataLoader(s2_vds, batch_size=S2_BATCH, **get_loader_kwargs(False))

        s2_loss = CombinedDeepSupLoss(4, DS_WEIGHTS,
                                       class_weight=MYO_CLASS_WEIGHT.to(device))
        # Differential learning rates: encoder at 10% of decoder LR to preserve ImageNet features
        s2_opt  = torch.optim.AdamW([
            {"params": s2.encoder_params(), "lr": S2_LR * S2_ENC_LR_MULT},
            {"params": s2.decoder_params(), "lr": S2_LR},
        ], weight_decay=S2_WD)

        swa_ep = int(S2_EPOCHS * SWA_START_FRAC)
        # OneCycleLR: warmup to max_lr then cosine anneal down; limited to epochs before SWA.
        # pct_start=0.10 allocates 10% of the schedule to warmup, reducing early gradient noise.
        # max_lr is 5× the base LR; div_factor and final_div_factor define start and end LRs.
        s2_sch = torch.optim.lr_scheduler.OneCycleLR(
            s2_opt,
            max_lr=[S2_LR * S2_ENC_LR_MULT * 5, S2_LR * 5],
            epochs=max(1, swa_ep - 1),  # scheduler only covers the pre-SWA phase
            steps_per_epoch=len(s2_tl),
            pct_start=0.10,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100)

        s2, s2d = train_stage(
            s2, s2_tl, s2_vl, s2_loss, s2_opt, s2_sch, device, fold_dir,
            S2_EPOCHS, S2_PAT, nc=4, name=f"s2_f{fi}",
            use_swa=True, swa_start_ep=swa_ep, swa_lr=SWA_LR)
        all_s2.append(copy.deepcopy(s2.cpu()))

        fold_results.append({"fold": fi, "s1_dice": s1d, "s2_dice": s2d})
        print(f"\n  Fold {fi + 1} | S1: {s1d:.4f} | S2: {s2d:.4f}")
        if _GLOBAL_CACHE: print(f"  {_GLOBAL_CACHE.stats()}")

    df_f = pd.DataFrame(fold_results)
    df_f.to_csv(EXP_DIR / "fold_results.csv", index=False)

    print(f"\n{'=' * 70}")
    print(df_f.to_string(index=False))
    print(f"\nMean S1: {df_f['s1_dice'].mean():.4f} ± {df_f['s1_dice'].std():.4f}")
    print(f"Mean S2: {df_f['s2_dice'].mean():.4f} ± {df_f['s2_dice'].std():.4f}")
    print(f"\nBaseline (v2_fast): ~0.905 mDice")
    print(f"This experiment:    {df_f['s2_dice'].mean():.4f}")
    print(f"{'=' * 70}")
    print(f"\n✓ Done! Run: {EXP_DIR}")
    print(f"\nNext steps:")
    print(f"  acdc_inference_best.py:")
    print(f"    EXP_DIR      = ACDC_ROOT / 'runs' / '{RUN_NAME}'")
    print(f"    CROP_MARGIN  = {CROP_MARGIN}  ← must match the value used during training")
    print(f"    S2_BBOX_STRAT = 's1_only'")
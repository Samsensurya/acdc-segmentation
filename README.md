# Two-Stage Cascaded Cardiac MRI Segmentation
### Ensemble Localization and Attention-Gated Decoding for ACDC

**Group 13** | Deep Learning for Medical Image Analysis | University of Twente  
Fabian Krüger · Ali Al Balushi · Joan Prenafeta · [Co-Author 4]

---

## Results

| Class | Dice | HD (mm) | ASSD (mm) |
|-------|------|---------|-----------|
| RV    | 0.908 | 11.58 | 0.71 |
| MYO   | 0.890 | 8.50  | 0.39 |
| LV    | 0.936 | 7.19  | 0.48 |
| **Mean** | **0.911** | **9.09** | **0.53** |

Surpasses Rank 6 on the ACDC leaderboard. Clinical parameters: r ≥ 0.991 for LVEDV, RVEDV, and LVEF.

---

## Method Overview

We propose a two-stage cascaded pipeline for automatic segmentation of the RV, MYO, and LV from cardiac cine MRI.

**Stage 1 — Heart Localisation**
- Lightweight MONAI U-Net trained on 256×256 slices with 2.5D three-channel input
- Five fold models combined into a **union bounding box** for robust localisation
- Threshold at 0.4, bounding box expanded by 40px; fallback to 160×160 central crop

**Stage 2 — Fine-Grained Segmentation**
- ResNet34 encoder (ImageNet pretrained) + Attention U-Net decoder
- Input: 192×192 cropped region from Stage 1
- Loss: Dice + Focal + 0.2 × approximate Hausdorff distance
- Deep supervision at three decoder stages (d2, d3, d4)
- Training: OneCycleLR for 87 epochs → SWA at constant lr = 5×10⁻⁵

**Inference**
- Five-model ensemble + four-fold test-time augmentation (identity, H-flip, V-flip, both)
- 3D connected-component post-processing (largest component per class)

---

## Repository Structure

```
Final ACDC/
├── database/                         # ACDC dataset (not included, see below)
├── runs/                             # Training checkpoints and logs
├── ACDC_DiceCE_INSTANCE_AUG.ipynb   # Baseline 2D U-Net experiments
├── acdc_eval.py                      # Evaluation: DSC, HD, ASSD, clinical metrics
├── acdc_inference.py                 # Inference script for test set
└── last_run_training.py              # Full training pipeline (Stage 1 + Stage 2)
```

---

## Setup

**Requirements**
```bash
pip install torch torchvision monai timm scipy nibabel numpy
```
Tested with Python 3.10 · PyTorch 2.1 · MONAI 1.3

**Dataset**

Download the ACDC dataset from the [official challenge page](https://acdc.creatis.insa-lyon.fr) and place it under `database/` with the following structure:

```
database/
├── training/
│   ├── patient001/
│   │   ├── patient001_4d.nii.gz
│   │   ├── patient001_frame01.nii.gz
│   │   └── patient001_frame01_gt.nii.gz
│   └── ...
└── testing/
    ├── patient101/
    └── ...
```

---

## Training

**Stage 1 — Localisation**
```bash
python last_run_training.py --stage 1
```

**Stage 2 — Segmentation**
```bash
python last_run_training.py --stage 2
```

Both stages use 5-fold cross-validation with patient-level splits (80 train / 20 val per fold).

---

## Inference & Evaluation

```bash
# Run inference on test set
python acdc_inference.py --data_dir database/testing --output_dir predictions/

# Evaluate predictions
python acdc_eval.py --pred_dir predictions/ --gt_dir database/testing
```

---

## Preprocessing Summary

- Intensity clipping: 1st–99th percentile + z-score normalisation
- In-plane resampling to 1.37 mm/pixel (dataset median) — bilinear for images, nearest-neighbour for masks
- No through-plane resampling (slice thickness varies 5–10 mm)
- Fixed 256×256 canvas via centre-pad / centre-crop

---

## Baseline

A single-stage 2D U-Net baseline (`ACDC_DiceCE_INSTANCE_AUG.ipynb`) was trained first to identify failure modes. Best validation Dice: **0.8705** (LV: 0.867, RV: 0.708, MYO: 0.670). The clear gap for RV and MYO motivated the cascaded design.

---

## Citation

If you use this code, please cite the ACDC challenge paper:

```
Bernard et al. Deep learning techniques for automatic MRI cardiac 
multi-structures segmentation and diagnosis: is the problem solved? 
IEEE Transactions on Medical Imaging, 37(11):2514–2525, 2018.
```

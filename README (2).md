# Two-Stage Cascaded Cardiac MRI Segmentation
### Ensemble Localization and Attention-Gated Decoding for ACDC

**Group 13** | Deep Learning for Medical Image Analysis | University of Twente  
Fabian Krüger · Ali Al Balushi · Joan Prenafeta · Samsensurya Selvaraj

---

## Overview

This repository contains the code for our submission to the
[Automatic Cardiac Diagnosis Challenge (ACDC)](https://acdc.creatis.insa-lyon.fr).
We propose a two-stage cascaded pipeline for automatic segmentation of the
right ventricle (RV), myocardium (MYO), and left ventricle (LV) from cardiac cine MRI.

A lightweight U-Net first localises the heart region; a ResNet34-based Attention U-Net
then segments the cropped region. Both stages use 2.5D three-channel input to capture
inter-slice context. Five Stage-1 fold models are combined into a union bounding box
for robust localisation, reducing the risk of any single model failing on pathologically
deformed hearts.

---

## Method

### Preprocessing
- Intensity clipping to 1st–99th percentile + z-score normalisation
- In-plane resampling to **1.37 mm/pixel** (dataset median)
- No through-plane resampling (slice thickness varies 5–10 mm)
- Fixed **256×256** canvas via centre-pad / centre-crop

### Stage 1 — Heart Localisation
- MONAI U-Net with channel widths (32, 64, 128, 256)
- Loss: Dice + cross-entropy | Optimiser: AdamW (lr=1e-3) | Schedule: cosine annealing
- **Union bounding box** from five fold models, expanded by 40px
- Fallback to fixed 160×160 central crop if no foreground detected

### Stage 2 — Fine-Grained Segmentation
- **ResNet34 encoder** (ImageNet pretrained) + **Attention U-Net decoder**
- Input: 192×192 crop from Stage 1 bounding box
- Loss: `L = L_Dice + L_Focal + 0.2 × L_HD`
- Deep supervision at decoder stages d2, d3, d4
- Schedule: **OneCycleLR** (first 35% of epochs) → **SWA** at constant lr=5e-5

### Inference
- Stage-1 ensemble + union bounding box
- Stage-2 ensemble with **4-fold TTA** (identity, H-flip, V-flip, both)
- **3D connected-component post-processing** — largest component per class

---

## Results

On the 50-patient ACDC test set (100 ED+ES frames):

| Class | Dice | HD (mm) | ASSD (mm) |
|-------|------|---------|-----------|
| RV    | 0.908 ± 0.063 | 11.58 ± 5.50 | 0.71 ± 0.72 |
| MYO   | 0.890 ± 0.026 | 8.50 ± 4.24  | 0.39 ± 0.24 |
| LV    | 0.936 ± 0.053 | 7.19 ± 4.64  | 0.48 ± 0.71 |
| **Mean** | **0.911** | **9.09** | **0.53** |

Clinical metrics: r ≥ 0.991 for LVEDV, RVEDV, and LVEF | LVEF bias: −0.14%  
Surpasses Rank 6 on the ACDC leaderboard.

---

## Repository Structure

```
├── acdc_training.py         # Full two-stage training pipeline (Stage 1 + Stage 2)
├── acdc_preprocessing.py    # Preprocessing pipeline: exports ACDC slices to .npy
│                            # Run this before acdc_training.py if starting fresh
├── acdc_inference.py        # Inference script for the test set
├── acdc_eval.py             # Evaluation: DSC, HD, ASSD, clinical metrics + figures
└── README.md
```

> `database/` (patient data) and `runs/` (model checkpoints) are excluded from this repository.

---

## Setup

```bash
pip install torch torchvision monai timm scipy nibabel numpy pandas medpy tqdm
```

Tested with Python 3.10 · PyTorch 2.1 · MONAI 1.3

---

## Data

Download the ACDC dataset from the
[official challenge page](https://acdc.creatis.insa-lyon.fr) and place it under `database/`:

```
database/
├── training/
│   ├── patient001/
│   │   ├── patient001_frame01.nii.gz
│   │   └── patient001_frame01_gt.nii.gz
│   └── ...        # patients 001–100
└── testing/
    ├── patient101/
    └── ...        # patients 101–150
```

The dataset contains 150 patients across 5 pathology groups: NOR, DCM, HCM, MINF, ARV.  
First 100 patients are used for training (5-fold CV), last 50 for testing.

---

## Usage

> **Before running any script**, open it and update the `ACDC_ROOT` and `EXP_DIR`
> path variables at the top to point to your local directory.
> These are hardcoded and will cause an error if not changed.

**1. Preprocessing** *(only needed once)*

In `acdc_preprocessing.py`, set:
```python
ACDC_ROOT = Path("/your/path/to/ACDC")
```
Then run:
```bash
python acdc_preprocessing.py
```
Exports all training slices to `database/prepared_2d_v2/`.

**2. Training**

In `acdc_training.py`, set:
```python
ACDC_ROOT = Path("/your/path/to/ACDC")
RUN_NAME  = None  # auto-generates a timestamped name on first run
                  # hardcode the generated name here when resuming
```
Then run:
```bash
python acdc_training.py
```
Checkpoints are saved under `runs/RUN_NAME/fold_*/checkpoints/`.

**3. Inference**

In `acdc_inference.py`, set:
```python
ACDC_ROOT = Path("/your/path/to/ACDC")
EXP_DIR   = ACDC_ROOT / "runs" / "your_run_name"
```
Then run:
```bash
python acdc_inference.py
```
Predictions are saved as NIfTI files under `EXP_DIR/predictions_test/`.

**4. Evaluation**

In `acdc_eval.py`, set:
```python
ACDC_ROOT = Path("/your/path/to/ACDC")
EXP_DIR   = ACDC_ROOT / "runs" / "your_run_name"
```
Then run:
```bash
python acdc_eval.py
```
Results (CSVs + figures) are saved under `EXP_DIR/evaluation/`.

---

## References

1. Bernard et al. *Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis.* IEEE TMI, 2018.
2. Baumgartner et al. *An exploration of 2D and 3D deep learning techniques for cardiac MR image segmentation.* STACOM, 2017.
3. Oktay et al. *Attention U-Net: learning where to look for the pancreas.* MIDL, 2018.
4. Karimi & Salcudean. *Reducing the Hausdorff distance in medical image segmentation.* IEEE TMI, 2020.
5. Izmailov et al. *Averaging weights leads to wider optima and better generalization.* UAI, 2018.

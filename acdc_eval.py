#!/usr/bin/env python3
"""
ACDC Evaluation — Segmentation and Clinical Metrics

Evaluates predictions (with 3D CCA post-processing applied) against ground
truth on train and/or test splits. Produces figures and CSVs covering:
  - Segmentation: Dice, Hausdorff distance, ASSD per class and phase
  - Clinical: LVEDV, RVEDV, LVEF, RVEF, MYMass (BSA-normalized)
  - Figures: bar charts, violin plots, group breakdown, Bland-Altman,
             scatter plots, qualitative overlays, poster summary table
"""

from __future__ import annotations
import re, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import (zoom, label as nd_label, binary_erosion,
                            distance_transform_edt)
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from medpy.metric.binary import dc as medpy_dc

warnings.filterwarnings("ignore")

# %% Configuration
ACDC_ROOT      = Path("/Users/fabian/Desktop/ACDC")
EXP_DIR        = ACDC_ROOT / "runs" / "last_run_20260331_115106"

PRED_DIR_TEST  = EXP_DIR / "predictions_test"
PRED_DIR_TRAIN = EXP_DIR / "predictions_train"
GT_DIR_TEST    = ACDC_ROOT / "database" / "testing"
GT_DIR_TRAIN   = ACDC_ROOT / "database" / "training"
OUT_DIR        = EXP_DIR / "evaluation"

EVAL_TEST  = True
EVAL_TRAIN = False

LABELS           = {1: "RV", 2: "MYO", 3: "LV"}
MYO_DENSITY_G_ML = 1.05   # g/mL, standard myocardial tissue density
VOXEL_TO_ML      = 1e-3   # mm³ → mL

ACCENT_COLOR     = "#4C72B0"
POSTER_DPI       = 200
REPORT_DPI       = 150

# %% Patient metadata
def read_info_cfg(patient_dir: Path) -> Dict[str, str]:
    cfg = {}
    p = patient_dir / "Info.cfg"
    if not p.exists(): return cfg
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def get_phase(patient_dir: Path, frame_name: str) -> str:
    """Map frame number to ED or ES using the patient Info.cfg."""
    cfg = read_info_cfg(patient_dir)
    ed = int(cfg.get("ED", 1)); es = int(cfg.get("ES", 2))
    m = re.search(r"frame(\d+)", frame_name)
    if m:
        fn = int(m.group(1))
        if fn == ed: return "ED"
        if fn == es: return "ES"
    return "unknown"


def get_group(patient_dir: Path) -> str:
    cfg = read_info_cfg(patient_dir)
    return cfg.get("Group", "unknown").upper()


def get_bsa(patient_dir: Path) -> float:
    """DuBois formula: BSA = 0.007184 × H^0.725 × W^0.425 (m²)."""
    cfg = read_info_cfg(patient_dir)
    try:
        h = float(cfg.get("Height", 170))
        w = float(cfg.get("Weight", 70))
        return 0.007184 * (h ** 0.725) * (w ** 0.425)
    except Exception:
        return 1.73  # population mean fallback

# %% Segmentation metrics
def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice similarity coefficient via medpy, consistent with official ACDC scoring."""
    if gt.sum() == 0 and pred.sum() == 0: return 1.0
    if gt.sum() == 0 or pred.sum() == 0:  return 0.0
    return float(medpy_dc(pred.astype(bool), gt.astype(bool)))


def compute_hd(pred: np.ndarray, gt: np.ndarray,
               voxel_spacing: Tuple) -> float:
    """Full Hausdorff distance (mm): max of both directed surface-to-surface distances.

    Surface voxels extracted via XOR with morphological erosion.
    HD = max( max(d_pred→gt), max(d_gt→pred) ).
    """
    pred_b = pred.astype(bool); gt_b = gt.astype(bool)
    if pred_b.sum() == 0 or gt_b.sum() == 0: return float("nan")
    try:
        dt_pred   = distance_transform_edt(~pred_b, sampling=voxel_spacing)
        dt_gt     = distance_transform_edt(~gt_b,   sampling=voxel_spacing)
        pred_surf = pred_b ^ binary_erosion(pred_b)
        gt_surf   = gt_b   ^ binary_erosion(gt_b)
        h_p2g = dt_gt[pred_surf]
        h_g2p = dt_pred[gt_surf]
        if len(h_p2g) == 0 or len(h_g2p) == 0: return float("nan")
        return float(max(np.max(h_p2g), np.max(h_g2p)))
    except Exception:
        return float("nan")


def compute_assd(pred: np.ndarray, gt: np.ndarray,
                 voxel_spacing: Tuple) -> float:
    """Average Symmetric Surface Distance (mm): mean of both directed distances."""
    pred_b = pred.astype(bool); gt_b = gt.astype(bool)
    if pred_b.sum() == 0 or gt_b.sum() == 0: return float("nan")
    try:
        pred_surf = pred_b ^ binary_erosion(pred_b)
        gt_surf   = gt_b   ^ binary_erosion(gt_b)
        dt_pred   = distance_transform_edt(~pred_b, sampling=voxel_spacing)
        dt_gt     = distance_transform_edt(~gt_b,   sampling=voxel_spacing)
        return float((dt_gt[pred_surf].mean() + dt_pred[gt_surf].mean()) / 2)
    except Exception:
        return float("nan")


def compute_volume_ml(mask: np.ndarray, voxel_spacing: Tuple) -> float:
    voxel_vol_mm3 = float(np.prod(voxel_spacing))
    return float(mask.astype(bool).sum() * voxel_vol_mm3 * VOXEL_TO_ML)

# %% Post-processing
def apply_3d_cca(vol: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Retain only the largest 3D connected component per foreground class.
    Removes isolated fragment false positives from ambiguous apical/basal slices.
    """
    out = np.zeros_like(vol)
    for c in range(1, num_classes):
        binary = (vol == c).astype(np.uint8)
        if binary.sum() == 0: continue
        labeled, n = nd_label(binary)
        if n == 0: continue
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        out[labeled == (np.argmax(sizes) + 1)] = c
    return out

# %% Per-volume evaluation
def evaluate_volume(pred_vol: np.ndarray, gt_vol: np.ndarray,
                    voxel_spacing: Tuple) -> Dict:
    """Compute Dice, HD, ASSD and volume per class after CCA post-processing."""
    pred_vol = apply_3d_cca(pred_vol)

    row = {}
    for label, name in LABELS.items():
        pc = (pred_vol == label); gc = (gt_vol == label)
        row[f"dice_{name}"] = compute_dice(pc, gc)
        row[f"hd_{name}"]   = compute_hd(pc, gc, voxel_spacing)
        row[f"assd_{name}"] = compute_assd(pc, gc, voxel_spacing)
        row[f"vol_{name}"]  = compute_volume_ml(pc, voxel_spacing)
        row[f"vol_{name}_gt"] = compute_volume_ml(gc, voxel_spacing)

    row["mean_dice_fg"] = float(np.mean([row[f"dice_{n}"] for n in LABELS.values()]))
    row["mean_hd_fg"]   = float(np.nanmean([row[f"hd_{n}"]   for n in LABELS.values()]))
    row["mean_assd_fg"] = float(np.nanmean([row[f"assd_{n}"] for n in LABELS.values()]))
    return row

# %% Clinical metrics
def compute_clinical_metrics(df: pd.DataFrame,
                              patient_dirs: Dict[str, Path]) -> pd.DataFrame:
    """Derive LVEDV, RVEDV, LVEF, RVEF, MYMass from ED and ES frames.
    All volumes normalized by BSA (mL/m²); myocardial mass in g/m².
    """
    clinical = []
    for pat in df["patient"].unique():
        df_p  = df[df["patient"] == pat]
        df_ed = df_p[df_p["phase"] == "ED"]
        df_es = df_p[df_p["phase"] == "ES"]
        if df_ed.empty or df_es.empty: continue

        r_ed = df_ed.iloc[0]; r_es = df_es.iloc[0]
        bsa  = get_bsa(patient_dirs[pat]) if pat in patient_dirs else 1.73

        def ef(edv, esv): return float((edv - esv) / edv * 100) if edv > 0 else float("nan")

        clinical.append({
            "patient":       pat,
            "LVEDV_ml_m2":   r_ed["vol_LV"]    / bsa,
            "LVEDV_gt_ml_m2":r_ed["vol_LV_gt"] / bsa,
            "RVEDV_ml_m2":   r_ed["vol_RV"]    / bsa,
            "RVEDV_gt_ml_m2":r_ed["vol_RV_gt"] / bsa,
            "LVEF_pct":      ef(r_ed["vol_LV"],    r_es["vol_LV"]),
            "LVEF_gt_pct":   ef(r_ed["vol_LV_gt"], r_es["vol_LV_gt"]),
            "RVEF_pct":      ef(r_ed["vol_RV"],    r_es["vol_RV"]),
            "RVEF_gt_pct":   ef(r_ed["vol_RV_gt"], r_es["vol_RV_gt"]),
            "MYMass_g_m2":   r_ed["vol_MYO"]    * MYO_DENSITY_G_ML / bsa,
            "MYMass_gt_g_m2":r_ed["vol_MYO_gt"] * MYO_DENSITY_G_ML / bsa,
        })

    return pd.DataFrame(clinical)

# %% Bland-Altman helper
def bland_altman_stats(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Bias, LoA (±1.96 σ), MAE, and Pearson r."""
    diff = pred - gt; mean_val = (pred + gt) / 2
    bias = float(np.nanmean(diff)); std = float(np.nanstd(diff))
    corr = float(np.corrcoef(pred[~np.isnan(pred)], gt[~np.isnan(gt)])[0, 1])
    mae  = float(np.nanmean(np.abs(diff)))
    return {"corr": corr, "bias": bias, "std": std, "mae": mae,
            "loa_low": bias - 1.96 * std, "loa_high": bias + 1.96 * std,
            "n": int(np.sum(~np.isnan(diff))),
            "mean_val": mean_val, "diff": diff}

# %% Figure 1 — Dice / HD / ASSD per class
def fig1_metrics_by_class(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    metrics_info = [
        ("dice", "Dice",        (0, 1.05)),
        ("hd",   "HD (mm)",     None),
        ("assd", "ASSD (mm)",   None),
    ]
    x = np.arange(len(LABELS)); w = 0.5

    for ax, (met, ylabel, ylim) in zip(axes, metrics_info):
        means = [df[f"{met}_{n}"].mean() for n in LABELS.values()]
        stds  = [df[f"{met}_{n}"].std()  for n in LABELS.values()]
        bars = ax.bar(x, means, w, yerr=stds, capsize=5,
                      color=ACCENT_COLOR, alpha=0.85)
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(list(LABELS.values()))
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
        if ylim: ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 2 — metrics per class × phase
def fig2_metrics_by_phase(df: pd.DataFrame, out_path: Path):
    phases = ["ED", "ES"]
    metrics_info = [("dice", "Dice"), ("hd", "HD (mm)"), ("assd", "ASSD (mm)")]
    fig, axes = plt.subplots(len(metrics_info), len(phases),
                              figsize=(11, 4 * len(metrics_info)))
    x = np.arange(len(LABELS)); w = 0.5

    for row_i, (met, ylabel) in enumerate(metrics_info):
        for col_i, phase in enumerate(phases):
            ax   = axes[row_i][col_i]
            df_p = df[df["phase"] == phase]
            means = [df_p[f"{met}_{n}"].mean() for n in LABELS.values()]
            stds  = [df_p[f"{met}_{n}"].std()  for n in LABELS.values()]
            bars = ax.bar(x, means, w, yerr=stds, capsize=5,
                          color=ACCENT_COLOR, alpha=0.85)
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{ylabel} — {phase}", fontweight="bold")
            ax.set_xticks(x); ax.set_xticklabels(list(LABELS.values()))
            ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 3 — Bland-Altman plots
def fig3_bland_altman(df_clin: pd.DataFrame, out_path: Path):
    metrics_clin = [
        ("LVEDV_ml_m2",  "LVEDV_gt_ml_m2",  "LVEDV (mL/m²)"),
        ("RVEDV_ml_m2",  "RVEDV_gt_ml_m2",  "RVEDV (mL/m²)"),
        ("LVEF_pct",     "LVEF_gt_pct",      "LVEF (%)"),
        ("RVEF_pct",     "RVEF_gt_pct",      "RVEF (%)"),
        ("MYMass_g_m2",  "MYMass_gt_g_m2",   "MYMass (g/m²)"),
    ]
    fig, axes = plt.subplots(1, len(metrics_clin),
                              figsize=(5 * len(metrics_clin), 4))

    for ax, (pred_col, gt_col, label) in zip(axes, metrics_clin):
        gt   = df_clin[gt_col].values.astype(float)
        pred = df_clin[pred_col].values.astype(float)
        mask = ~(np.isnan(gt) | np.isnan(pred))
        s = bland_altman_stats(pred[mask], gt[mask])
        ax.scatter(s["mean_val"], s["diff"], s=22, color=ACCENT_COLOR, alpha=0.7)
        ax.axhline(s["bias"],     color="b", lw=1.2, ls="--",
                   label=f"bias = {s['bias']:.2f}")
        ax.axhline(s["loa_low"],  color="b", lw=0.8, ls=":", alpha=0.7,
                   label=f"LoA [{s['loa_low']:.1f}, {s['loa_high']:.1f}]")
        ax.axhline(s["loa_high"], color="b", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Mean of GT and Pred")
        ax.set_ylabel("Pred − GT")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 4 — clinical metrics table (CSV)
def fig4_clinical_table(df_clin: pd.DataFrame, out_path: Path):
    metrics_clin = [
        ("LVEDV_ml_m2",  "LVEDV_gt_ml_m2",  "LVEDV_ml_m2"),
        ("RVEDV_ml_m2",  "RVEDV_gt_ml_m2",  "RVEDV_ml_m2"),
        ("LVEF_pct",     "LVEF_gt_pct",      "LVEF_pct"),
        ("RVEF_pct",     "RVEF_gt_pct",      "RVEF_pct"),
        ("MYMass_g_m2",  "MYMass_gt_g_m2",   "MYMass_g_m2"),
    ]
    rows = []
    for pred_col, gt_col, metric_name in metrics_clin:
        gt   = df_clin[gt_col].values.astype(float)
        pred = df_clin[pred_col].values.astype(float)
        mask = ~(np.isnan(gt) | np.isnan(pred))
        s = bland_altman_stats(pred[mask], gt[mask])
        rows.append({"metric": metric_name, "corr": s["corr"], "bias": s["bias"],
                     "std": s["std"], "mae": s["mae"],
                     "loa_low": s["loa_low"], "loa_high": s["loa_high"], "n": s["n"]})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(df_out[["metric", "corr", "bias", "std", "mae"]].to_string(index=False))
    return df_out

# %% Figure 5 — scatter plots (GT vs prediction)
def fig5_scatter_clinical(df_clin: pd.DataFrame, out_path: Path):
    """Identity-line scatter with Pearson r annotation per clinical metric."""
    metrics_clin = [
        ("LVEDV_ml_m2",  "LVEDV_gt_ml_m2",  "LVEDV (mL/m²)"),
        ("RVEDV_ml_m2",  "RVEDV_gt_ml_m2",  "RVEDV (mL/m²)"),
        ("LVEF_pct",     "LVEF_gt_pct",      "LVEF (%)"),
        ("RVEF_pct",     "RVEF_gt_pct",      "RVEF (%)"),
        ("MYMass_g_m2",  "MYMass_gt_g_m2",   "MYMass (g/m²)"),
    ]
    fig, axes = plt.subplots(1, len(metrics_clin),
                              figsize=(5 * len(metrics_clin), 4.5))

    for ax, (pred_col, gt_col, label) in zip(axes, metrics_clin):
        gt   = df_clin[gt_col].values
        pred = df_clin[pred_col].values
        mask = ~(np.isnan(gt) | np.isnan(pred))
        ax.scatter(gt[mask], pred[mask], s=25, color=ACCENT_COLOR, alpha=0.7)
        mn = min(gt[mask].min(), pred[mask].min())
        mx = max(gt[mask].max(), pred[mask].max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.6)
        r = np.corrcoef(gt[mask], pred[mask])[0, 1]
        ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, va="top")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Ground truth"); ax.set_ylabel("Prediction")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 6 — mean metrics by phase (line plot)
def fig6_mean_metrics_by_phase(df: pd.DataFrame, out_path: Path):
    phases = ["ED", "ES"]
    metrics_info = [
        ("mean_dice_fg",  "Mean Dice (foreground)"),
        ("mean_hd_fg",    "Mean HD (mm)"),
        ("mean_assd_fg",  "Mean ASSD (mm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = [axes[0][0], axes[0][1], axes[1][0]]
    axes[1][1].set_visible(False)

    for ax, (met, ylabel) in zip(axes_flat, metrics_info):
        vals = [df[df["phase"] == p][met].mean() for p in phases]
        ax.plot(phases, vals, "o-", color=ACCENT_COLOR, lw=2)
        for ph, v in zip(phases, vals):
            ax.annotate(f"{v:.3f}", (ph, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_ylabel(ylabel); ax.set_xlabel("Phase")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 7 — violin plots of Dice distribution per class
def fig7_violin_dice(df: pd.DataFrame, out_path: Path):
    """Dice distribution per class as violin plots.
    More informative than mean bars: exposes inter-patient variability and
    bimodal distributions caused by failure cases.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for ax, name in zip(axes, LABELS.values()):
        data = df[f"dice_{name}"].dropna().values
        parts = ax.violinplot([data], positions=[1],
                              showmedians=True, showextrema=True)
        parts["bodies"][0].set_facecolor(ACCENT_COLOR); parts["bodies"][0].set_alpha(0.7)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color("black"); parts[key].set_linewidth(1.0)

        med = np.median(data)
        ax.text(1, med + 0.02, f"med = {med:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(name, fontweight="bold")
        ax.set_xticks([]); ax.set_ylabel("Dice"); ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Dice score distribution per class", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 8 — Dice per pathology group
def fig8_dice_by_group(df: pd.DataFrame, out_path: Path):
    """Mean Dice per ACDC pathology group (NOR, DCM, HCM, MINF, RV) split by phase.
    Reveals which pathologies are hardest to segment.
    """
    groups = sorted(df["group"].dropna().unique())
    class_names = list(LABELS.values())
    x = np.arange(len(groups)); width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, phase in zip(axes, ["ED", "ES"]):
        df_phase = df[df["phase"] == phase]
        for i, (name, color) in enumerate(zip(class_names, colors)):
            vals = [df_phase[df_phase["group"] == g][f"dice_{name}"].mean()
                    for g in groups]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"Dice by pathology group — {phase}", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("Dice"); ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 9 — qualitative segmentation overlays
def fig9_qualitative_overlays(pred_dir: Path, gt_dir: Path, out_path: Path,
                               n_examples: int = 4):
    """Mid-slice MRI | GT | prediction overlays for n_examples ED frames.
    Colors: RV=red, MYO=green, LV=blue (semi-transparent).
    """
    # collect ED frames that have both image and GT available
    ed_files = []
    for pf in sorted(pred_dir.glob("*_pred.nii.gz")):
        patient_id  = pf.name.split("_frame")[0]
        frame_name  = pf.name.replace("_pred.nii.gz", "")
        patient_dir = gt_dir / patient_id
        if get_phase(patient_dir, frame_name) != "ED": continue
        gt_path  = patient_dir / f"{frame_name}_gt.nii.gz"
        img_path = patient_dir / f"{frame_name}.nii.gz"
        if gt_path.exists() and img_path.exists():
            ed_files.append((pf, gt_path, img_path))
        if len(ed_files) >= n_examples: break

    if not ed_files:
        print("  Qualitative overlay: no matching ED frames found, skipping.")
        return

    n = len(ed_files)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1: axes = axes[np.newaxis, :]

    cmap_colors = [(0, 0, 0, 0), (0.9, 0.15, 0.15, 0.55),
                   (0.15, 0.85, 0.15, 0.55), (0.15, 0.35, 0.9, 0.55)]
    seg_cmap = ListedColormap(cmap_colors)

    for row_i, (pred_path, gt_path, img_path) in enumerate(ed_files):
        img_vol  = nib.load(str(img_path)).get_fdata().astype(np.float32)
        gt_vol   = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
        pred_vol = apply_3d_cca(nib.load(str(pred_path)).get_fdata().astype(np.uint8))

        z_mid = img_vol.shape[2] // 2   # middle slice for representative anatomy
        img_sl  = img_vol[..., z_mid]
        lo, hi  = np.percentile(img_sl, 1), np.percentile(img_sl, 99)
        img_norm = np.clip((img_sl - lo) / (hi - lo + 1e-8), 0, 1)

        patient_id = pred_path.name.split("_frame")[0]
        for col_i, (seg, title) in enumerate([(None,                 "MRI"),
                                               (gt_vol[..., z_mid],  "Ground truth"),
                                               (pred_vol[..., z_mid],"Prediction")]):
            ax = axes[row_i][col_i]
            ax.imshow(img_norm.T, cmap="gray", origin="lower")
            if seg is not None:
                ax.imshow(seg.T, cmap=seg_cmap, vmin=0, vmax=3, origin="lower")
            ax.set_title(f"{patient_id} — {title}", fontsize=9)
            ax.axis("off")

    legend_patches = [mpatches.Patch(color=cmap_colors[1][:3], label="RV"),
                      mpatches.Patch(color=cmap_colors[2][:3], label="MYO"),
                      mpatches.Patch(color=cmap_colors[3][:3], label="LV")]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=POSTER_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Figure 10 — poster summary table
def fig10_poster_summary(df: pd.DataFrame, out_path: Path):
    """Rendered metrics table (mean ± std) for poster use — no LaTeX required."""
    rows = []
    for phase in ["ED", "ES", "All"]:
        df_p = df[df["phase"] == phase] if phase != "All" else df
        for name in LABELS.values():
            rows.append({
                "Phase": phase, "Class": name,
                "Dice":      f"{df_p[f'dice_{name}'].mean():.3f} ± {df_p[f'dice_{name}'].std():.3f}",
                "HD (mm)":   f"{df_p[f'hd_{name}'].mean():.2f} ± {df_p[f'hd_{name}'].std():.2f}",
                "ASSD (mm)": f"{df_p[f'assd_{name}'].mean():.2f} ± {df_p[f'assd_{name}'].std():.2f}",
            })

    df_tbl = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, len(rows) * 0.5 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=df_tbl.values, colLabels=df_tbl.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)

    for col_i in range(len(df_tbl.columns)):
        tbl[(0, col_i)].set_facecolor("#2C4770")
        tbl[(0, col_i)].set_text_props(color="white", fontweight="bold")
    for row_i in range(1, len(rows) + 1):
        shade = "#EEF2F8" if row_i % 2 == 0 else "white"
        for col_i in range(len(df_tbl.columns)):
            tbl[(row_i, col_i)].set_facecolor(shade)

    ax.set_title("Segmentation performance (mean ± std)", fontsize=12,
                 fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=POSTER_DPI, bbox_inches="tight")
    plt.close(fig)

# %% Main evaluation loop
def run_evaluation(pred_dir: Path, gt_dir: Path, split_name: str):
    out = OUT_DIR / split_name
    out.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*_pred.nii.gz"))
    if not pred_files:
        print(f"  WARNING: no predictions found in {pred_dir}")
        return

    print(f"\n{'=' * 65}")
    print(f"  EVALUATING: {split_name.upper()} ({len(pred_files)} frames)")
    print(f"{'=' * 65}")

    all_rows = []; patient_dirs = {}

    for pred_path in pred_files:
        patient_id  = pred_path.name.split("_frame")[0]
        frame_name  = pred_path.name.replace("_pred.nii.gz", "")
        patient_dir = gt_dir / patient_id
        gt_path     = patient_dir / f"{frame_name}_gt.nii.gz"

        if not gt_path.exists():
            print(f"  WARNING: GT missing for {frame_name}"); continue

        pred_nii = nib.load(str(pred_path)); gt_nii = nib.load(str(gt_path))
        pred_vol = pred_nii.get_fdata().astype(np.uint8)
        gt_vol   = gt_nii.get_fdata().astype(np.uint8)
        spacing  = tuple(float(x) for x in gt_nii.header.get_zooms()[:3])

        if pred_vol.shape != gt_vol.shape:
            zf = tuple(g / p for g, p in zip(gt_vol.shape, pred_vol.shape))
            pred_vol = zoom(pred_vol.astype(np.float32), zf, order=0).astype(np.uint8)

        row = evaluate_volume(pred_vol, gt_vol, spacing)
        row.update({"patient": patient_id, "frame": frame_name,
                    "phase": get_phase(patient_dir, frame_name),
                    "group": get_group(patient_dir)})
        all_rows.append(row)
        patient_dirs[patient_id] = patient_dir

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "segmentation_metrics.csv", index=False)

    df_clin = compute_clinical_metrics(df, patient_dirs)
    df_clin.to_csv(out / "clinical_metrics.csv", index=False)

    print(f"\n  Generating figures...")
    fig1_metrics_by_class(df,              out / "fig1_metrics_by_class.png")
    fig2_metrics_by_phase(df,              out / "fig2_metrics_by_phase.png")
    fig6_mean_metrics_by_phase(df,         out / "fig6_mean_metrics_by_phase.png")
    fig7_violin_dice(df,                   out / "fig7_violin_dice.png")
    fig8_dice_by_group(df,                 out / "fig8_dice_by_group.png")
    fig9_qualitative_overlays(pred_dir, gt_dir, out / "fig9_qualitative_overlays.png")
    fig10_poster_summary(df,               out / "fig10_poster_summary.png")

    if not df_clin.empty:
        fig5_scatter_clinical(df_clin,     out / "fig5_scatter_clinical.png")
        fig3_bland_altman(df_clin,         out / "fig3_bland_altman.png")
        print(f"\n  Clinical metrics summary:")
        fig4_clinical_table(df_clin,       out / "fig4_clinical_stats.csv")

    # console summary
    print(f"\n  SEGMENTATION SUMMARY")
    print(f"  {'':5s}  {'Dice':>8}  {'HD (mm)':>9}  {'ASSD (mm)':>10}")
    for name in LABELS.values():
        print(f"  {name:5s}  {df[f'dice_{name}'].mean():>8.4f}  "
              f"{df[f'hd_{name}'].mean():>9.2f}  {df[f'assd_{name}'].mean():>10.2f}")
    print(f"  {'Mean':5s}  {df['mean_dice_fg'].mean():>8.4f}  "
          f"{df['mean_hd_fg'].mean():>9.2f}  {df['mean_assd_fg'].mean():>10.2f}")

    print(f"\n  Per-phase Dice:")
    for phase in ["ED", "ES"]:
        df_p = df[df["phase"] == phase]
        if df_p.empty: continue
        vals = [f"{df_p[f'dice_{n}'].mean():.4f}" for n in LABELS.values()]
        print(f"    {phase}: RV={vals[0]}  MYO={vals[1]}  LV={vals[2]}")

    print(f"\n  Figures saved to: {out}")

# %% Main
if __name__ == "__main__":
    if "XXXXXX" in str(EXP_DIR):
        raise ValueError("EXP_DIR is still a placeholder — set it to the training run directory.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if EVAL_TEST:
        if not PRED_DIR_TEST.exists():
            raise FileNotFoundError(f"Test predictions not found: {PRED_DIR_TEST}")
        run_evaluation(PRED_DIR_TEST, GT_DIR_TEST, split_name="test")

    if EVAL_TRAIN:
        if not PRED_DIR_TRAIN.exists():
            print(f"WARNING: train predictions not found: {PRED_DIR_TRAIN}")
        else:
            run_evaluation(PRED_DIR_TRAIN, GT_DIR_TRAIN, split_name="train")

    print(f"\n✓ Done. All results saved to: {OUT_DIR}")